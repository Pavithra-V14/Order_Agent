"""
Regression test for a REAL bug found in production, not in this
sandbox: on a real, already-running Postgres instance whose mock_orders
and exception_cases tables predated this session's new columns,
init_db() (Base.metadata.create_all()) silently did nothing for those
already-existing tables - create_all() only creates tables that don't
exist yet, it never alters an existing one. Every one of the seven demo
scripts then failed with psycopg2.errors.UndefinedColumn on the very
first query touching mock_orders.

This never surfaced in this sandbox's own test suite because every
test here runs against a freshly created SQLite file - the table
doesn't exist yet either, so create_all() genuinely creates it with
every current column, masking the exact gap a real, persistent
database hits.
"""
import os
import tempfile

from sqlalchemy import create_engine, text, inspect


def test_ensure_new_columns_adds_missing_columns_to_a_pre_existing_table():
    """Reproduces the actual failure directly: creates the OLD schema
    (tables that predate payment_fingerprint/working_memory_summary,
    exactly as a real already-running database would have), then
    proves _ensure_new_columns() adds them without requiring the table
    to be dropped and recreated."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_migration_{os.getpid()}_{id(object())}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}")

    try:
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE mock_orders (order_id VARCHAR PRIMARY KEY, customer_id VARCHAR, "
                "total_amount_usd FLOAT)"
            ))
            conn.execute(text(
                "CREATE TABLE exception_cases (id VARCHAR PRIMARY KEY, order_id VARCHAR)"
            ))

        import app.core.db as db_module
        original_engine = db_module.engine
        db_module.engine = engine
        try:
            db_module._ensure_new_columns()

            inspector = inspect(engine)
            order_cols = {c["name"] for c in inspector.get_columns("mock_orders")}
            case_cols = {c["name"] for c in inspector.get_columns("exception_cases")}
            assert "payment_fingerprint" in order_cols, (
                "the real production failure: mock_orders.payment_fingerprint must be added "
                "to an already-existing table, not just present on freshly created ones"
            )
            assert "working_memory_summary" in case_cols
        finally:
            db_module.engine = original_engine
    finally:
        engine.dispose()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ensure_new_columns_is_a_safe_noop_when_columns_already_exist():
    """Idempotency: calling this on every startup (init_db() does) must
    not error or duplicate anything once the columns are already there
    - covers both a freshly create_all()'d database and a second call
    on an already-migrated one."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_migration_noop_{os.getpid()}_{id(object())}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}")

    try:
        import app.core.db as db_module
        original_engine = db_module.engine
        db_module.engine = engine
        try:
            db_module.Base.metadata.create_all(bind=engine)  # fresh DB - already has every current column
            db_module._ensure_new_columns()  # must not raise
            db_module._ensure_new_columns()  # calling it AGAIN must also not raise or duplicate

            inspector = inspect(engine)
            order_cols = [c["name"] for c in inspector.get_columns("mock_orders")]
            assert order_cols.count("payment_fingerprint") == 1
        finally:
            db_module.engine = original_engine
    finally:
        engine.dispose()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ensure_new_columns_skips_a_table_that_does_not_exist_at_all():
    """A table that doesn't exist yet at all (not even the old schema)
    is create_all()'s job, not this function's - must not try to ALTER
    a nonexistent table."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_migration_notable_{os.getpid()}_{id(object())}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}")  # completely empty database, no tables at all

    try:
        import app.core.db as db_module
        original_engine = db_module.engine
        db_module.engine = engine
        try:
            db_module._ensure_new_columns()  # must not raise even though neither table exists
        finally:
            db_module.engine = original_engine
    finally:
        engine.dispose()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_init_db_calls_ensure_new_columns():
    """Wiring check: init_db() (called by every demo script and
    app.main on startup) must actually invoke the migration, not just
    have it exist as a callable nobody calls - exactly the class of gap
    this project's own memory audit found repeatedly elsewhere."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_migration_wiring_{os.getpid()}_{id(object())}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path}")

    try:
        import app.core.db as db_module
        original_engine = db_module.engine
        db_module.engine = engine
        try:
            called = {"n": 0}
            original_fn = db_module._ensure_new_columns
            db_module._ensure_new_columns = lambda: called.update(n=called["n"] + 1)
            try:
                db_module.init_db()
            finally:
                db_module._ensure_new_columns = original_fn
            assert called["n"] == 1
        finally:
            db_module.engine = original_engine
    finally:
        engine.dispose()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
