"""Standalone image-gallery packaging: exact original bytes, scope and offline scripts."""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("contact_bundle", Path(__file__).resolve().parents[1] / "build_contact_web_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class GalleryBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / "trial"
        self.trial.mkdir()
        self.manifest = self.root / "manifest.json"
        self.catalog = {
            "version": 1,
            "groups": [{"id": "current", "title": "当前图表", "archived": False}, {"id": "history", "title": "历史版本", "archived": True}],
            "items": [{"id": "first", "group": "current", "path": "first.png", "title": "接触对比", "description": "原图", "archived": False}],
        }
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.catalog, ensure_ascii=False), encoding="utf-8")

    def image(self, relative):
        path = self.trial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        return path

    def test_includes_nested_unlisted_images_and_preserves_bytes(self):
        self.image("first.png")
        self.image("review/other.png")
        self.image("contact_explorer_preview.png")
        self.image("contact_explorer_gallery_preview.png")
        result = MODULE.gallery_data(self.trial, self.manifest)
        self.assertEqual([item["path"] for item in result["items"]], ["first.png", "review/other.png"])
        for item in result["items"]:
            self.assertEqual(base64.b64decode(item["src"].split(",", 1)[1]), PNG)
            self.assertEqual((item["width"], item["height"]), (1, 1))
            self.assertFalse(item["archived"])
        self.assertEqual(result["totalBytes"], 2 * len(PNG))

    def test_history_from_group_or_directory_is_explicit(self):
        self.image("first.png")
        self.image("previous_run/legacy.png")
        self.catalog["items"][0]["group"] = "history"
        self.save_manifest()
        result = MODULE.gallery_data(self.trial, self.manifest)
        self.assertTrue(all(item["archived"] for item in result["items"]))
        self.assertTrue(all(group["archived"] for group in result["groups"]))

    def test_excluded_groups_are_not_reimported_as_unlisted_images(self):
        self.image("first.png")
        self.image("history/old.png")
        self.image("unlisted.png")
        self.catalog["items"].append({"id": "old", "group": "history", "path": "history/old.png", "title": "旧图"})
        self.catalog["excludedGroups"] = ["history"]
        self.catalog["includeUnlisted"] = False
        self.save_manifest()
        result = MODULE.gallery_data(self.trial, self.manifest)
        self.assertEqual([item["path"] for item in result["items"]], ["first.png"])
        self.assertEqual([group["id"] for group in result["groups"]], ["current"])
        self.assertTrue((self.trial / "history/old.png").exists(), "original files are retained")

    def test_rejects_out_of_scope_paths_and_links(self):
        self.catalog["items"][0]["path"] = "../private.png"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "评估目录内"):
            MODULE.gallery_data(self.trial, self.manifest)
        self.catalog["items"][0]["path"] = "first.png"
        self.save_manifest()
        outside = self.root / "private.png"
        outside.write_bytes(PNG)
        (self.trial / "first.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "超出评估目录"):
            MODULE.gallery_data(self.trial, self.manifest)

    def test_rejects_duplicate_ids(self):
        self.catalog["items"].append(dict(self.catalog["items"][0]))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "重复"):
            MODULE.gallery_data(self.trial, self.manifest)

    def test_bundle_orders_scripts_after_body_and_escapes_script_text(self):
        source = self.root / "source"
        source.mkdir()
        names = ["data.js", "app.js", "gallery-data.js", "gallery.js"]
        source.joinpath("index.html").write_text('<html><head><link rel="stylesheet" href="style.css">' + ''.join(f'<script src="{name}" defer></script>' for name in names) + '</head><body><main id="ready"></main></body></html>', encoding="utf-8")
        source.joinpath("style.css").write_text("body{background:white}", encoding="utf-8")
        for i, name in enumerate(names):
            source.joinpath(name).write_text(f"window.sequence{i} = true;", encoding="utf-8")
        self.image("first.png")
        self.catalog["items"][0]["description"] = "</script><script>unsafe()</script></SCRIPT></ScRiPt>"
        self.save_manifest()
        html = MODULE.bundle(source, self.trial, self.manifest)
        self.assertNotIn('src="', html)
        self.assertNotIn('<link ', html)
        self.assertEqual(html.count('<script>\n'), 4)
        self.assertNotIn('</script><script>unsafe()', html)
        self.assertNotIn('</SCRIPT>', html)
        self.assertNotIn('</ScRiPt>', html)
        self.assertIn('<\\/script>', html)
        self.assertIn('<\\/SCRIPT>', html)
        self.assertLess(html.index('id="ready"'), html.index('window.sequence0'))
        self.assertLess(html.index('window.sequence1'), html.index('window.CONTACT_GALLERY'))
        self.assertLess(html.index('window.CONTACT_GALLERY'), html.index('window.sequence3'))

    def test_optional_synced_assets_are_embedded_after_player(self):
        source = self.root / "sync_source"
        source.mkdir()
        scripts = ["data.js", "app.js", "details-data.js", "timeline.js", "media-data.js", "media.js"]
        styles = ["style.css", "timeline.css", "media.css"]
        source.joinpath("index.html").write_text('<html><head>' + ''.join(f'<link rel="stylesheet" href="{name}">' for name in styles) + ''.join(f'<script src="{name}" defer></script>' for name in scripts) + '</head><body><main></main></body></html>', encoding="utf-8")
        for i, name in enumerate(scripts):
            source.joinpath(name).write_text(f"window.module{i} = true;", encoding="utf-8")
        for i, name in enumerate(styles):
            source.joinpath(name).write_text(f".module{i}{{color:black}}", encoding="utf-8")
        html = MODULE.bundle(source)
        self.assertNotIn('src="', html)
        self.assertNotIn('<link ', html)
        self.assertEqual(html.count('<style>'), 3)
        self.assertEqual(html.count('<script>\n'), 6)
        self.assertEqual(sorted(html.index(f'window.module{i}') for i in range(6)), [html.index(f'window.module{i}') for i in range(6)])

    def test_report_p95_preserves_original_values_and_missing_channels(self):
        report = {"channels": {
            "finger_thumb_normal_n": {"thresholds": {"expert_contact_p95_n": 3.5256361825389515}},
            "finger_index_normal_n": {"thresholds": {"expert_contact_p95_n": 0}},
            "finger_middle_normal_n": {"thresholds": {"expert_contact_p95_n": True}},
            "grip_normal_n": {"phases": {"board_contact": {"reference": {"p95_n": 7.094952051387747}, "rollout": {"p95_n": float("nan")}}}},
        }}
        (self.trial / "contact_intensity.json").write_text(json.dumps(report), encoding="utf-8")
        result = MODULE.gallery_metrics_data(self.trial)
        self.assertEqual(result["fingerP95N"], [3.5256361825389515, 0, None, None, None])
        self.assertEqual(result["phaseP95N"]["grip_normal_n"]["board_contact"], {"reference": 7.094952051387747, "rollout": None})
        self.assertEqual(result["phaseP95N"]["board_normal_n"], {})
        json.dumps(result, allow_nan=False)

    def test_missing_report_does_not_invent_statistics(self):
        self.assertEqual(MODULE.gallery_metrics_data(self.trial), {"version": 1, "fingerP95N": [None] * 5, "phaseP95N": {}})

    def test_report_symlink_must_stay_within_trial(self):
        outside = self.root / "private_report.json"
        outside.write_text("{}", encoding="utf-8")
        (self.trial / "contact_intensity.json").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "超出评估目录"):
            MODULE.gallery_metrics_data(self.trial)

    def test_live_gallery_modules_follow_data_and_registry_producers(self):
        source = self.root / "live_source"
        source.mkdir()
        scripts = ["data.js", "app.js", "details-data.js", "timeline.js", "media-data.js", "media.js", "gallery-metrics-data.js", "live-spatial.js", "live-metrics.js", "live-gallery.js"]
        styles = ["style.css", "live-gallery.css", "live-spatial.css", "live-metrics.css"]
        source.joinpath("index.html").write_text('<html><head>' + ''.join(f'<link rel="stylesheet" href="{name}">' for name in styles) + ''.join(f'<script src="{name}" defer></script>' for name in scripts) + '</head><body><main></main></body></html>', encoding="utf-8")
        for i, name in enumerate(scripts):
            source.joinpath(name).write_text(f"window.liveModule{i} = true;", encoding="utf-8")
        for i, name in enumerate(styles):
            source.joinpath(name).write_text(f".liveModule{i}{{color:black}}", encoding="utf-8")
        html = MODULE.bundle(source, self.trial)
        self.assertNotIn('src="', html)
        self.assertNotIn('<link ', html)
        self.assertEqual(html.count('<style>'), 4)
        self.assertEqual(html.count('<script>\n'), 10)
        self.assertLess(html.index('window.liveModule5'), html.index('window.CONTACT_GALLERY_METRICS'))
        self.assertLess(html.index('window.CONTACT_GALLERY_METRICS'), html.index('window.liveModule7'))
        self.assertLess(html.index('window.liveModule7'), html.index('window.liveModule8'))
        self.assertLess(html.index('window.liveModule8'), html.index('window.liveModule9'))

    def test_explicit_trial_data_does_not_reuse_template_examples(self):
        source = self.root / "template"
        source.mkdir()
        assets = self.root / "seed000_data"
        assets.mkdir()
        names = ["data.js", "app.js", "details-data.js", "timeline.js", "media-data.js", "media.js"]
        source.joinpath("index.html").write_text('<html><head><link rel="stylesheet" href="style.css">' + ''.join(f'<script src="{name}" defer></script>' for name in names) + '</head><body></body></html>', encoding="utf-8")
        source.joinpath("style.css").write_text("body{}", encoding="utf-8")
        for name in names:
            source.joinpath(name).write_text(f'window.template = "{name}";', encoding="utf-8")
        for name in ("data.js", "details-data.js", "media-data.js"):
            assets.joinpath(name).write_text(f'window.seed000 = "{name}";', encoding="utf-8")
        html = MODULE.bundle(source, data_dir=assets)
        for name in ("data.js", "details-data.js", "media-data.js"):
            self.assertIn(f'window.seed000 = "{name}";', html)
            self.assertNotIn(f'window.template = "{name}";', html)
        self.assertEqual(source.joinpath("data.js").read_text(), 'window.template = "data.js";')
        assets.joinpath("media-data.js").unlink()
        with self.assertRaises(FileNotFoundError):
            MODULE.bundle(source, data_dir=assets)


if __name__ == "__main__":
    unittest.main()
