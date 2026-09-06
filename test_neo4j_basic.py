"""
Bare-minimum, standalone Neo4j connection test. No dependency on this
project at all - just the neo4j driver. Run directly:

    pip install neo4j
    python3 test_neo4j_basic.py

Fill in your own URI/user/password below first.
"""
from neo4j import GraphDatabase

URI = "neo4j+s://YOUR-INSTANCE-ID.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "YOUR-PASSWORD"


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
    test_normal()
    test_insecure_diagnostic_only()
