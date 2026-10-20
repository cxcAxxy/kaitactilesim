"""Real temporary TCP sockets only; no model, evaluator, or persistent service."""

import errno
import importlib.util
from pathlib import Path
import socket
from unittest.mock import patch

import pytest
from kaihand_tactile_env.pipeline import deployment

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate/evaluate_pi05_stage_batch.py"
SPEC = importlib.util.spec_from_file_location("pi05_stage_batch_ports_test", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def leave_server_time_wait(host):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = server.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    return port


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_time_wait_is_reusable_without_stopping_any_process(host):
    port = leave_server_time_wait(host)
    # Reproduce the old false positive: no live listener, but a plain bind
    # fails because the accepted server connection remains in TIME_WAIT.
    with socket.socket() as plain_probe:
        with pytest.raises(OSError) as caught:
            plain_probe.bind(("127.0.0.1", port))
        assert caught.value.errno == errno.EADDRINUSE
    assert batch.port_free(port)
    deployment._assert_managed_endpoint_available(f"ws://127.0.0.1:{port}", bind_host=host)
    # The actual service's address-reuse policy can also bind and listen.
    with socket.socket() as next_server:
        next_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        next_server.bind(("0.0.0.0", port))
        next_server.listen(1)
        assert not batch.port_free(port)
        with pytest.raises(deployment.DeploymentError, match="already in use"):
            deployment._assert_managed_endpoint_available(f"ws://127.0.0.1:{port}", bind_host=host)
    assert batch.port_free(port)


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("reuse", [False, True])
def test_live_listener_is_never_treated_as_reusable(host, reuse):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse))
        listener.bind((host, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert not batch.port_free(port)
        with pytest.raises(deployment.DeploymentError, match="already in use"):
            deployment._assert_managed_endpoint_available(f"ws://127.0.0.1:{port}", bind_host="0.0.0.0")
        # The probe must neither stop the listener nor consume a connection.
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            connection, _ = listener.accept()
            connection.close()


@pytest.mark.parametrize("phase", ["bind", "listen"])
@pytest.mark.parametrize("code", [errno.EADDRINUSE, errno.EACCES, errno.EPERM, errno.EADDRNOTAVAIL])
def test_managed_endpoint_reports_the_actual_socket_error(phase, code):
    error = OSError(code, "Socket check failed")
    message = "already in use" if code == errno.EADDRINUSE else "cannot check managed server endpoint"
    with patch.object(deployment.socket, "socket") as factory:
        probe = factory.return_value.__enter__.return_value
        getattr(probe, phase).side_effect = error
        with pytest.raises(deployment.DeploymentError, match=message) as caught:
            deployment._assert_managed_endpoint_available("ws://127.0.0.1:18980", bind_host="0.0.0.0")
        assert caught.value.__cause__ is error
        probe.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
