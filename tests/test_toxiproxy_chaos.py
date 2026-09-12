"""
Real Toxiproxy-based chaos testing - architecture doc Layer 16 calls
for this specifically ("Chaos testing (Toxiproxy) on circuit-breaker/
retry logic ... release-blocking"), which this project's existing
chaos tests (tests/test_phase8_execution.py) never actually used -
those simulate failure via gateway.inject_transient_failures(), a
Python-level in-process exception, not genuine network-level fault
injection. Both are valuable, but they test different things: the
existing tests prove the circuit breaker's STATE MACHINE logic is
correct; this proves it behaves correctly against REAL network chaos
(a reset connection, not a mocked exception) - the actual thing
Toxiproxy exists for.

HONEST LIMITATION, found directly while building this, distinct from
this project's other documented network limitations (no outbound route
to api.stripe.com/api.groq.com from the sandbox): a REAL toxiproxy-server
v2.9.0 binary WAS successfully downloaded from Toxiproxy's actual GitHub
releases and DOES run correctly (confirmed: `toxiproxy-server --version`
prints "toxiproxy-server version 2.9.0" via a direct, non-backgrounded
invocation) - but starting it, or any listening server, as a background
process in this specific sandbox consistently and immediately kills the
entire invoking shell with no output, across multiple different process-
management approaches (nohup, setsid, subprocess.Popen with full fd
detachment), even under a hard `timeout` wrapper. This appears to be a
sandbox-level restriction on opening listening sockets, not a bug in
this test's own code or in Toxiproxy itself.

Given that, this test deliberately does NOT attempt to start
toxiproxy-server itself (the risky operation that caused the hangs) -
it only checks whether one is ALREADY reachable, with a short, bounded
timeout, and skips cleanly if not. On a real machine (which almost
certainly does not have this sandbox's specific listening-socket
restriction), start toxiproxy-server once yourself and this suite runs
for real:
    # download once: https://github.com/Shopify/toxiproxy/releases
    ./toxiproxy-server &
    pytest tests/test_toxiproxy_chaos.py -v
"""
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

TOXIPROXY_API_URL = "http://127.0.0.1:8474"


def _toxiproxy_already_running() -> bool:
    """A short, hard-bounded check - deliberately does NOT attempt to
    start toxiproxy-server itself, since that specific operation is
    what caused this sandbox to kill the invoking shell outright
    during development. Only ever checks for an ALREADY-running
    instance, safely and quickly."""
    import httpx
    try:
        httpx.get(f"{TOXIPROXY_API_URL}/version", timeout=1.0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _toxiproxy_already_running(),
    reason=f"No toxiproxy-server reachable at {TOXIPROXY_API_URL}. This test suite deliberately never "
           f"tries to start one itself (see this file's module docstring for why) - start "
           f"./toxiproxy-server yourself first (download: "
           f"https://github.com/Shopify/toxiproxy/releases), then re-run.",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OKHandler(BaseHTTPRequestHandler):
    """A trivial real HTTP server standing in for a downstream
    dependency (payment/carrier gateway) - Toxiproxy sits in front of
    THIS, not a mock, so the chaos it injects is genuine network-level
    disruption of a real TCP connection."""
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass  # keep test output clean


@pytest.fixture
def real_backend_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _OKHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture
def toxiproxy_client():
    """Connects to an ALREADY-running toxiproxy-server (see the module
    docstring for why this deliberately never tries to start one
    itself)."""
    from toxiproxy import Toxiproxy
    client = Toxiproxy()
    client.destroy_all()  # clean slate - no leftover proxies from a previous run

    yield client

    client.destroy_all()


def test_circuit_breaker_trips_on_genuine_network_resets(real_backend_server, toxiproxy_client):
    """THE real chaos test: a real TCP connection reset (not a
    monkeypatched Python exception) must genuinely cause httpx to
    raise, must genuinely count as a circuit-breaker failure, and the
    breaker must genuinely trip OPEN after real_failure_threshold real
    network failures - proven against actual network-level fault
    injection, not simulated in-process behavior."""
    import httpx
    from app.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError

    proxy_port = _free_port()
    proxy = toxiproxy_client.create(
        name="test-real-chaos-proxy",
        listen=f"127.0.0.1:{proxy_port}",
        upstream=f"127.0.0.1:{real_backend_server}",
    )
    proxy.add_toxic(type="reset_peer", name="reset", attributes={"timeout": 0})

    proxy_url = f"http://127.0.0.1:{proxy_port}"
    breaker = CircuitBreaker(name="test-real-chaos", failure_threshold=3, reset_timeout_seconds=30.0)

    def real_network_call():
        # A genuine httpx GET against the toxic proxy - if this
        # succeeds, the toxic isn't actually injecting real chaos and
        # this test would be meaningless.
        resp = httpx.get(proxy_url, timeout=2.0)
        resp.raise_for_status()
        return resp

    real_failures = 0
    for _ in range(3):
        try:
            breaker.call(real_network_call)
        except Exception:
            real_failures += 1

    assert real_failures == 3, (
        f"expected all 3 calls to genuinely fail against the real reset_peer toxic, got {real_failures} failures"
    )
    assert breaker.state == CircuitState.OPEN, "the breaker must trip OPEN after real, genuine network failures"
    assert breaker.call_attempts == 3

    # Fail-fast proof: a 4th call must NOT attempt the real network
    # call at all - CircuitOpenError raised immediately, not another
    # real connection attempt.
    with pytest.raises(CircuitOpenError):
        breaker.call(real_network_call)
    assert breaker.call_attempts == 3, "a 4th call while OPEN must not increment call_attempts at all"


def test_circuit_breaker_recovers_after_real_chaos_toxic_is_removed(real_backend_server, toxiproxy_client):
    """The other half: once the real network disruption is REMOVED
    (the toxic is disabled, simulating the dependency recovering), the
    breaker must genuinely recover on the next successful real call -
    proving this isn't a one-way trip, tested against real behavior."""
    import httpx
    from app.core.circuit_breaker import CircuitBreaker, CircuitState

    proxy_port = _free_port()
    proxy = toxiproxy_client.create(
        name="test-real-chaos-recovery-proxy",
        listen=f"127.0.0.1:{proxy_port}",
        upstream=f"127.0.0.1:{real_backend_server}",
    )
    toxic = proxy.add_toxic(type="reset_peer", name="reset", attributes={"timeout": 0})

    proxy_url = f"http://127.0.0.1:{proxy_port}"
    breaker = CircuitBreaker(name="test-real-chaos-recovery", failure_threshold=2, reset_timeout_seconds=0.1)

    def real_network_call():
        resp = httpx.get(proxy_url, timeout=2.0)
        resp.raise_for_status()
        return resp

    for _ in range(2):
        try:
            breaker.call(real_network_call)
        except Exception:
            pass
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.15)  # past reset_timeout_seconds - breaker allows a real HALF_OPEN probe next

    proxy.destroy_toxic("reset")  # the real dependency "recovers" - genuinely removes the network chaos

    result = breaker.call(real_network_call)  # a real, genuinely successful network call this time
    assert result.status_code == 200
    assert breaker.state == CircuitState.CLOSED, "a real successful call after recovery must close the breaker"
