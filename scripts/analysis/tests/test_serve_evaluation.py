"""Report serving, address restrictions, and confinement without opening sockets."""
from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve_evaluation
from serve_evaluation import (
    EvaluationRequestHandler,
    parse_byte_range,
    resolve_report_path,
)


class BindHostTests(unittest.TestCase):
    def test_explicit_loopback_and_private_ipv4_addresses_are_allowed(self):
        for address in (
            "127.0.0.1", "127.0.0.0", "127.255.255.255",
            "10.0.0.0", "10.255.255.255",
            "172.16.0.0", "172.31.255.255",
            "192.168.0.0", "192.168.0.54", "192.168.255.255",
        ):
            with self.subTest(address=address):
                self.assertEqual(serve_evaluation.parse_bind_host(address), address)

    def test_wildcard_public_and_other_non_private_addresses_are_rejected(self):
        for address in (
            "0.0.0.0", "8.8.8.8", "22.52.244.197",
            "172.15.255.255", "172.32.0.0", "192.167.255.255", "192.169.0.0",
            "169.254.0.1", "224.0.0.1", "255.255.255.255",
            "localhost", "evaluation.example.com", "", "192.168.0.999",
            "::1", "fd00::1", "::ffff:192.168.0.54",
        ):
            with self.subTest(address=address):
                with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                    serve_evaluation.parse_bind_host(address)

    def test_main_binds_default_loopback_and_requested_private_address(self):
        with tempfile.TemporaryDirectory() as directory:
            for arguments, expected_host, expected_port in (
                ([], "127.0.0.1", 8876),
                (["--host", "192.168.0.54", "--port", "8877"], "192.168.0.54", 8877),
            ):
                with self.subTest(arguments=arguments):
                    with patch.object(serve_evaluation, "ThreadingHTTPServer") as constructor:
                        server = constructor.return_value.__enter__.return_value
                        server.serve_forever.side_effect = KeyboardInterrupt
                        output = StringIO()
                        with redirect_stdout(output):
                            result = serve_evaluation.main(["--directory", directory, *arguments])
                        self.assertEqual(result, 0)
                        constructor.assert_called_once()
                        address, handler = constructor.call_args.args
                        self.assertEqual(address, (expected_host, expected_port))
                        self.assertIs(handler.func, EvaluationRequestHandler)
                        self.assertEqual(handler.keywords["directory"], Path(directory).resolve())
                        server.serve_forever.assert_called_once_with()
                        self.assertIn(f"http://{expected_host}:{expected_port}/", output.getvalue())

    def test_invalid_host_is_rejected_before_creating_server(self):
        with tempfile.TemporaryDirectory() as directory:
            for address in ("0.0.0.0", "8.8.8.8", "22.52.244.197", "localhost", "::1"):
                with self.subTest(address=address):
                    with patch.object(serve_evaluation, "ThreadingHTTPServer") as constructor:
                        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                            serve_evaluation.main(["--directory", directory, "--host", address])
                        self.assertEqual(raised.exception.code, 2)
                        constructor.assert_not_called()


class RangeParserTests(unittest.TestCase):
    def test_closed_open_suffix_and_clamped_ranges(self):
        self.assertIsNone(parse_byte_range(None, 10))
        self.assertEqual(parse_byte_range("bytes=2-4", 10), (2, 4))
        self.assertEqual(parse_byte_range("bytes=2-", 10), (2, 9))
        self.assertEqual(parse_byte_range("bytes=-3", 10), (7, 9))
        self.assertEqual(parse_byte_range("bytes=-30", 10), (0, 9))
        self.assertEqual(parse_byte_range("bytes=0-30", 10), (0, 9))

    def test_bad_or_multiple_ranges_are_rejected(self):
        for header in ("bytes=", "bytes=-", "bytes=-0", "bytes=10-", "bytes=5-3", "bytes=0-1,3-4", "items=0-1"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_byte_range(header, 10)
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=0-", 0)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "report"
        self.root.mkdir()
        (self.root / "video.mp4").write_bytes(b"0123456789")
        (self.root / "index.html").write_text("report")

    def handler(self, header=None, path="/video.mp4"):
        handler = object.__new__(EvaluationRequestHandler)
        handler.directory, handler.path = self.root, path
        handler.headers = {} if header is None else {"Range": header}
        handler.wfile = BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()
        return handler

    def test_partial_get_returns_exact_body_and_headers(self):
        handler = self.handler("bytes=3-6")
        handler.do_GET()
        handler.send_response.assert_called_once_with(206)
        handler.send_header.assert_any_call("Content-Range", "bytes 3-6/10")
        handler.send_header.assert_any_call("Content-Length", "4")
        handler.send_header.assert_any_call("Accept-Ranges", "bytes")
        self.assertEqual(handler.wfile.getvalue(), b"3456")

    def test_head_range_has_headers_but_no_body(self):
        handler = self.handler("bytes=-2")
        handler.do_HEAD()
        handler.send_response.assert_called_once_with(206)
        handler.send_header.assert_any_call("Content-Range", "bytes 8-9/10")
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_full_get_and_empty_file_are_valid(self):
        handler = self.handler()
        handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue(), b"0123456789")
        (self.root / "empty").write_bytes(b"")
        handler = self.handler(path="/empty")
        handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call("Content-Length", "0")

    def test_invalid_range_returns_416_without_body(self):
        handler = self.handler("bytes=100-")
        handler.do_GET()
        handler.send_response.assert_called_once_with(416)
        handler.send_header.assert_any_call("Content-Range", "bytes */10")
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_index_query_and_external_symlink_confinement(self):
        self.assertEqual(resolve_report_path(self.root, "/?a=1"), self.root / "index.html")
        outside = self.root.parent / "outside"
        outside.write_bytes(b"private")
        (self.root / "escape").symlink_to(outside)
        for path in ("/escape", "/../outside", "/%2e%2e/outside"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                resolve_report_path(self.root, path)
        handler = self.handler(path="/escape")
        handler.do_GET()
        handler.send_error.assert_called_once_with(403, "Report path is not accessible")
        self.assertEqual(handler.wfile.getvalue(), b"")


if __name__ == "__main__":
    unittest.main()
