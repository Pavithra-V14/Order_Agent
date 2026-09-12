"""
Bare-minimum, standalone Neo4j connection test. No dependency on this
project at all - just the neo4j driver. Run directly:

    pip install neo4j
    python3 test_neo4j_basic.py

Fill in your own URI/user/password below first.
"""
from neo4j import GraphDatabase

URI="neo4j+ssc://3bee6dca.databases.neo4j.io"
USER="3bee6dca"
PASSWORD="iWqXMrJHbB8Yy0kQC0AN2wnufBdvtekOk-rRplm9NxI"

def test_normal():
    """The real, secure connection - same as your actual app will use."""
    print("=== Normal (verified TLS) connection ===")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    finally:
        driver.close()


async def test_async():
    """
    Tests ONLY the neo4j package's own ASYNC driver - zero Graphiti
    involvement at all. This isolates one specific question: is the
    sync-vs-async asymmetry (sync connects fine, async fails) a bug in
    the `neo4j` package itself on this machine/environment, or is it
    something specific to how Graphiti wraps/uses that driver?

    If this ALSO fails the same way test_normal() succeeds: the bug is
    in the neo4j async driver itself here (a real environment issue —
    Python version, network stack, uv's packaging, something below
    Graphiti entirely) — not something fixable in this project's
    Graphiti integration code at all.

    If this SUCCEEDS: the bug is specific to how Graphiti constructs or
    uses its Neo4jDriver wrapper, not the underlying neo4j package —
    narrows the next fix to Graphiti's own code, not this project's.
    """
    print("\n=== Async (neo4j package's own AsyncGraphDatabase, NO Graphiti) ===")
    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        await driver.verify_connectivity()
        print("SUCCESS (async, no Graphiti)")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    finally:
        await driver.close()


def test_insecure_diagnostic_only():
    """
    DIAGNOSTIC ONLY - never use this scheme in real code or leave it
    enabled. Swaps the URI scheme from neo4j+s:// (strict certificate
    validation) to neo4j+ssc:// (still encrypted, but trusts
    self-signed/unverified certificates) to answer ONE question: is the
    network path to Aura actually open, or is something blocking it
    entirely?

    (Note: neo4j+s:// already implies strict encryption on its own and
    will actively REJECT a separately-passed trusted_certificates/
    encrypted/ssl_context config option with a ConfigurationError -
    confirmed directly. Swapping the URI scheme itself is the correct,
    only way to change this behavior, not an extra parameter.)

    - If this SUCCEEDS while test_normal() FAILS with a certificate
      error: the network path is fine - something is actively
      substituting its own (self-signed) certificate for Aura's real
      one. That's TLS interception (VPN/antivirus/corporate proxy),
      confirmed, not a connectivity/firewall problem.
    - If this ALSO fails (e.g. with a timeout or connection-refused,
      not a cert error): the network path itself is blocked - a
      firewall/port issue, not a certificate one.
    """
    print("\n=== Insecure (neo4j+ssc://, skips cert verification) - DIAGNOSTIC ONLY ===")
    insecure_uri = URI.replace("neo4j+s://", "neo4j+ssc://")
    driver = GraphDatabase.driver(insecure_uri, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print("SUCCESS (with cert verification skipped)")
        print("-> This CONFIRMS TLS interception: the network path works fine,")
        print("   but something is presenting a fake certificate for Aura's real one.")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("-> The network path itself appears blocked, not a cert-trust issue.")
    finally:
        driver.close()


if __name__ == "__main__":
    import asyncio
    test_normal()
    asyncio.run(test_async())
    # test_insecure_diagnostic_only() is no longer needed — the sync test
    # already succeeded with FULL certificate verification (no TLS
    # interception), so that theory is already ruled out. Uncomment
    # below only if you want to re-confirm it:
    # test_insecure_diagnostic_only()
