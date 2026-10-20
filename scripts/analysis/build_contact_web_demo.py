"""将静态接触演示打包为可离线打开、可嵌入网站的单个 HTML。"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import re
import struct


IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def gallery_data(trial: Path, manifest: Path) -> dict:
    """收录原始图片；历史版本分组，网页自身截图不递归收录。"""
    root = trial.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"评估目录无效：{trial}")
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    excluded_groups = set(catalog.get("excludedGroups", []))
    groups = [dict(group) for group in catalog["groups"]]
    group_ids = [group["id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("图表分类 ID 重复")
    known = {}
    ids = set()
    for item in catalog["items"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("图表路径必须位于评估目录内")
        if item["group"] not in group_ids:
            raise ValueError(f"图表分类不存在：{item['group']}")
        if item["path"] in known or item["id"] in ids:
            raise ValueError("图表路径或 ID 重复")
        known[item["path"]] = dict(item)
        ids.add(item["id"])
    found = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name.startswith("contact_explorer") and "preview" in path.stem:
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"图片链接超出评估目录：{relative}")
        found[relative] = path

    items = []
    # Preserve curated ordering, then include unlisted images rather than losing them.
    paths = [path for path in known if path in found and known[path]["group"] not in excluded_groups]
    if catalog.get("includeUnlisted", True):
        paths += [path for path in found if path not in known]
    for relative in paths:
        path = found[relative]
        item = dict(known.get(relative, {}))
        if not item:
            archived = any(re.search(r"previous|backup|archive", part, re.I) for part in path.relative_to(root).parts[:-1])
            group_id = "other-history" if archived else "other-images"
            if group_id not in group_ids:
                groups.append({"id": group_id, "title": "其他历史图片" if archived else "其他图片", "description": "原始图片，解释与色标以图内标注为准。", "archived": archived})
                group_ids.append(group_id)
            item = {"id": f"file:{relative}", "path": relative, "group": group_id, "title": path.stem, "description": "评估目录中的原始可视化图片。", "archived": archived}
        group = next(group for group in groups if group["id"] == item["group"])
        item["archived"] = bool(item.get("archived", False) or group.get("archived", False))
        raw = path.read_bytes()
        item["src"] = f"data:{IMAGE_TYPES[path.suffix.lower()]};base64," + base64.b64encode(raw).decode("ascii")
        item["bytes"] = len(raw)
        if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
            item["width"], item["height"] = struct.unpack(">II", raw[16:24])
        items.append(item)
    active_groups = {item["group"] for item in items}
    return {"version": 1, "groups": [group for group in groups if group["id"] in active_groups], "items": items, "totalBytes": sum(item["bytes"] for item in items)}


def gallery_script(trial: Path, manifest: Path) -> str:
    return "window.CONTACT_GALLERY = " + json.dumps(gallery_data(trial, manifest), ensure_ascii=False, separators=(",", ":")) + ";"


def gallery_metrics_data(trial: Path) -> dict:
    """Copy full-episode contact P95 and full-phase P95, never recompute at 10 Hz."""
    root = trial.resolve(strict=True)
    report = root / "contact_intensity.json"
    result = {"version": 1, "fingerP95N": [None] * 5, "phaseP95N": {}}
    if not report.exists():
        return result
    if not report.resolve().is_relative_to(root):
        raise ValueError("统计报告链接超出评估目录")
    channels = json.loads(report.read_text(encoding="utf-8")).get("channels", {})

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    result["fingerP95N"] = [number(channels.get(f"finger_{finger}_normal_n", {}).get("thresholds", {}).get("expert_contact_p95_n")) for finger in ("thumb", "index", "middle", "ring", "pinky")]
    for channel in ("grip_normal_n", "board_normal_n"):
        result["phaseP95N"][channel] = {
            phase: {key: number(stats.get(key, {}).get("p95_n")) for key in ("reference", "rollout")}
            for phase, stats in channels.get(channel, {}).get("phases", {}).items()
        }
    return result


def bundle(source: Path, trial: Path | None = None, manifest: Path | None = None,
           data_dir: Path | None = None) -> str:
    html = (source / "index.html").read_text(encoding="utf-8")
    css_files = ["style.css"]
    for optional in ("timeline.css", "media.css", "live-gallery.css", "live-spatial.css", "live-metrics.css"):
        if re.search(r'href=["\'](?:\./)?' + re.escape(optional) + r'["\']', html):
            css_files.append(optional)
    for filename in css_files:
        css = (source / filename).read_text(encoding="utf-8")
        styles = re.compile(r'<link\b[^>]*href=["\'](?:\./)?' + re.escape(filename) + r'["\'][^>]*>', re.I)
        html, count = styles.subn(lambda _: "<style>\n" + css + "\n</style>", html)
        if count != 1:
            raise ValueError(f"预期唯一 {filename} 引用，实际 {count} 个")
    inline_scripts = []
    filenames = ["data.js", "app.js"]
    if re.search(r'src=["\'](?:\./)?gallery\.js["\']', html):
        filenames += ["gallery-data.js", "gallery.js"]
    if re.search(r'src=["\'](?:\./)?timeline\.js["\']', html):
        filenames += ["details-data.js", "timeline.js"]
    if re.search(r'src=["\'](?:\./)?media\.js["\']', html):
        filenames += ["media-data.js", "media.js"]
    if re.search(r'src=["\'](?:\./)?live-gallery\.js["\']', html):
        filenames += ["gallery-metrics-data.js", "live-spatial.js", "live-metrics.js", "live-gallery.js"]
    for filename in filenames:
        if filename == "gallery-data.js" and trial is not None:
            body = gallery_script(trial, manifest or source / "gallery-manifest.json")
        elif filename == "gallery-metrics-data.js" and trial is not None:
            body = "window.CONTACT_GALLERY_METRICS = " + json.dumps(gallery_metrics_data(trial), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";"
        else:
            asset_root = data_dir if data_dir is not None and filename in ("data.js", "details-data.js", "media-data.js") else source
            body = (asset_root / filename).read_text(encoding="utf-8")
        pattern = re.compile(r'<script\b[^>]*src=["\'](?:\./)?' + re.escape(filename) + r'["\'][^>]*>\s*</script>', re.I)
        escaped = re.sub(r"</script", lambda match: match.group(0).replace("</", "<\\/"), body, flags=re.I)
        html, count = pattern.subn("", html)
        if count != 1:
            raise ValueError(f"预期唯一 {filename} 引用，实际 {count} 个")
        inline_scripts.append("<script>\n" + escaped + "\n</script>")
    # Inline classic scripts do not honor defer. Run after the document body
    # exists, preserving data-before-application execution order.
    if html.lower().count("</body>") != 1:
        raise ValueError("预期唯一 body 结束标签")
    html = re.sub(r"</body>", lambda _: "\n".join(inline_scripts) + "\n</body>", html, flags=re.I)
    if re.search(r'<script\b[^>]*\bsrc=', html, re.I):
        raise ValueError("打包结果仍有外部脚本引用")
    if re.search(r'<link\b[^>]*\brel=["\']stylesheet["\']', html, re.I):
        raise ValueError("打包结果仍有外部样式引用")
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2] / "demos" / "contact-explorer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="本次轨迹的 data.js、details-data.js、media-data.js 目录；不改写模板的示例数据")
    parser.add_argument("--trial-dir", type=Path, help="按分类清单内嵌评估图片；不修改原图，不包含网页预览截图或已排除分组")
    parser.add_argument("--gallery-manifest", type=Path, help="图片中文说明和分类清单，默认读取源目录 gallery-manifest.json")
    args = parser.parse_args()
    content = bundle(args.source_dir, args.trial_dir, args.gallery_manifest, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(f"已生成独立 HTML：{args.output}（{args.output.stat().st_size / 1024 / 1024:.2f} MiB）")


if __name__ == "__main__":
    main()
