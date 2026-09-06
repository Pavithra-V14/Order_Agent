"""
Isolated Neo4j Aura connectivity test - tests ONLY the Neo4j connection,
completely independent of Graphiti/Groq, to pinpoint exactly where a
"SSLCertVerificationError: self-signed certificate in certificate chain"
or "Unable to retrieve routing information" error is actually coming from.

A self-signed-certificate error on a neo4j+s:// (Aura) connection almost
always means something BETWEEN your machine and Aura is intercepting the
TLS connection and presenting its own certificate instead of Aura's real
one - most commonly:
  - A corporate/office network or VPN doing SSL/TLS inspection
  - Antivirus software (Kaspersky, ESET, some Windows Defender configs)
    that intercepts HTTPS/TLS traffic to scan it
  - A firewall or proxy blocking port 7687 (Neo4j's Bolt port) and
    returning its own error page instead

It is NOT typically a problem with Neo4j Aura's own certificate - Aura
issues real, CA-signed certificates by default.

Usage:
    python3 scripts/test_neo4j_connection.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings


def main():
    settings = get_settings()
    if not settings.neo4j_uri:
        print("NEO4J_URI is not set in .env - nothing to test.")
        sys.exit(1)

    print(f"Testing connection to: {settings.neo4j_uri}")
    print(f"User: {settings.neo4j_user}")
    print()

    import neo4j
    try:
        driver = neo4j.GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=10.0,
        )
        driver.verify_connectivity()
        print("SUCCESS - Neo4j Aura is reachable and credentials are valid.")
        driver.close()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print()

        # The neo4j driver wraps the REAL underlying error inside a
        # generic ServiceUnavailable/"Unable to retrieve routing
        # information" message — that generic wrapper looks IDENTICAL
        # whether the actual cause is a blocked port or a TLS
        # interception, so printing only the top-level exception (as an
        # earlier version of this script did) hides the one detail that
        # actually tells them apart. Walking __cause__ surfaces it.
        full_chain = []
        cause = e
        while cause is not None:
            full_chain.append(f"{type(cause).__name__}: {cause}")
            cause = cause.__cause__
        print("Full exception chain (root cause is usually the LAST line):")
        for i, line in enumerate(full_chain):
            print(f"  [{i}] {line}")
        print()

        full_text = " ".join(full_chain)
        if "CERTIFICATE_VERIFY_FAILED" in full_text or "self-signed" in full_text:
            print("Root cause: TLS interception (VPN/antivirus/corporate proxy).")
            print("  1. Disable VPN/corporate proxy temporarily and re-run this script.")
            print("  2. Check antivirus 'HTTPS scanning'/'SSL inspection' - disable, re-test.")
            print("  3. Try from a different network entirely (e.g. mobile hotspot).")
            print("  4. Confirm NEO4J_URI starts with 'neo4j+s://' exactly, no typos.")
        elif "routing information" in full_text or "ServiceUnavailable" in full_text:
            print("Root cause: could not reach Aura at all (no TLS handshake attempted,")
            print("per the chain above having no certificate-related line). Try:")
            print("  1. A different network (mobile hotspot) - many office/ISP networks")
            print("     block port 7687 (Neo4j's Bolt port) outright.")
            print("  2. Confirm the instance is actually RUNNING in the Aura console")
            print("     (not paused - Aura free-tier instances auto-pause after inactivity).")
            print("  3. Double check NEO4J_URI has no typos (copy-paste directly from Aura).")
        sys.exit(1)


if __name__ == "__main__":
    main()
