#!/usr/bin/env python3
"""Read-only static report server for loopback or an explicit private IPv4."""
from __future__ import annotations

import argparse
import mimetypes
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from urllib.parse import unquote, urlsplit

_PRIVATE_NETWORKS = tuple(IPv4Network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
))


def parse_bind_host(value: str) -> str:
    """Accept one literal loopback/RFC1918 address, never all interfaces."""
    try:
        address = IPv4Address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--host requires a literal private or loopback IPv4 address") from error
    if not (address.is_loopback or any(address in network for network in _PRIVATE_NETWORKS)):
        raise argparse.ArgumentTypeError("--host must be loopback or RFC1918 private IPv4; public/all-interface binds are unsupported")
    return str(address)


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return inclusive byte bounds; unsupported/unsatisfiable ranges raise ValueError."""
    if size < 0:
        raise ValueError("negative file size")
    if header is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or not any(match.groups()) or size == 0:
        raise ValueError("invalid or unsatisfiable byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("empty suffix range")
        return max(0, size - length), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


def resolve_report_path(directory: Path, request_path: str) -> Path:
    """Resolve URL paths within the report directory, including symlink targets."""
    root = directory.resolve(strict=True)
    path = unquote(urlsplit(request_path).path, errors="strict")
    if "\x00" in path:
        raise ValueError("NUL in request path")
    resolved = (root / path.lstrip("/")).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PermissionError("path is outside the report directory")
    if resolved.is_dir():
        resolved = (resolved / "index.html").resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise PermissionError("index target is outside the report directory")
    if not resolved.is_file():
        raise FileNotFoundError("not a regular report file")
    return resolved


class EvaluationRequestHandler(BaseHTTPRequestHandler):
    """GET and HEAD only; no writes, uploads, or directory listing."""

    def __init__(self, *args, directory: str | Path, **kwargs):
        self.directory = Path(directory).resolve(strict=True)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def _serve(self, *, send_body: bool):
        try:
            path = resolve_report_path(self.directory, self.path)
            file = path.open("rb")
        except PermissionError:
            self.send_error(403, "Report path is not accessible")
            return
        except (FileNotFoundError, NotADirectoryError, OSError):
            self.send_error(404, "Report file not found")
            return
        except (ValueError, UnicodeError, RuntimeError):
            self.send_error(400, "Invalid report path")
            return
        with file:
            # fstat describes the file actually opened, not a later path lookup.
            import os
            stat = os.fstat(file.fileno())
            size = stat.st_size
            try:
                selected = parse_byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = selected if selected is not None else (0, size - 1)
            length = max(0, end - start + 1)
            self.send_response(206 if selected is not None else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if selected is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            file.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # Browser seeking can cancel a prior video request.
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Existing static report directory")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--host", type=parse_bind_host, default="127.0.0.1",
                        help="Explicit loopback or RFC1918 IPv4 address (default: 127.0.0.1)")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error("--directory must be an existing directory")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    handler = partial(EvaluationRequestHandler, directory=args.directory.resolve())
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Serving report at http://{args.host}:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
