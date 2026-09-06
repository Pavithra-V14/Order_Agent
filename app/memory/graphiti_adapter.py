"""
Production swap point for episodic.py -- Zep/Graphiti backed by Neo4j or
FalkorDB, per architecture doc 8.4. Not runnable in this sandbox (no
Docker, no network access to stand up a graph DB).

To activate in a real deployment:

    pip install graphiti-core
    # run Neo4j (docker-compose.yml already provisions postgres/redis/
    # qdrant -- add a neo4j service the same way) or FalkorDB

    from graphiti_core import Graphiti
    client = Graphiti(uri="bolt://localhost:7687", user="neo4j", password="...")

    # log_episode() in episodic.py becomes:
    await client.add_episode(
        name=episode_type,
        episode_body=json.dumps(content),
        source_description=f"case:{case_id}",
        reference_time=occurred_at,
        group_id=customer_id,
    )

    # get_customer_history() becomes a Graphiti search scoped to the
    # customer's group_id, which is where the real upgrade over the SQL
    # substitute shows up: Graphiti's search does semantic + graph-aware
    # retrieval, not just a time-ordered SQL filter, and supports
    # multi-hop queries (e.g. "entities related to this customer's
    # address") that episodic.py's SQL table structurally cannot do.

This module intentionally contains no runnable code -- it exists so the
swap point is documented in the codebase itself, not only in the
architecture doc, and so nobody mistakes episodic.py's SQL table for the
final intended implementation.
"""
