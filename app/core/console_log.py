"""
Structured console logging for agent/tool/RAG activity - visible,
real-time terminal output showing what's actually happening as the
system runs, distinct from the DB-persisted TraceSpanRecord data (which
is queryable after the fact but invisible while watching a terminal).

A real, previously-missing piece of operational visibility: without
this, running the app or a demo script gave no indication in the
terminal of which agents fired, which tools were called, or what RAG
retrieved - only the final result, with no way to watch the system
"think" without separately querying trace data afterward.
"""
import logging
import sys

_logger = logging.getLogger("order_exception_agent")

if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_agent_step(agent_name: str, message: str, case_id: str = None) -> None:
    """A named agent (diagnosis, fraud, inventory, customer_context,
    resolution_policy, execution) doing something worth seeing in the
    terminal in real time."""
    prefix = f"[case={case_id[:8]}] " if case_id else ""
    _logger.info(f"AGENT  {agent_name:<20} {prefix}{message}")


def log_tool_call(tool_name: str, is_write: bool, status: str, latency_ms: float = None, case_id: str = None) -> None:
    """A tool call (payment/wms/carrier/oms) completing - mirrors what
    record_tool_call() persists to the database, printed to the
    terminal at the same moment so it's visible live, not just
    queryable afterward."""
    prefix = f"[case={case_id[:8]}] " if case_id else ""
    kind = "WRITE" if is_write else "READ "
    latency_str = f"{latency_ms:.1f}ms" if latency_ms is not None else "?"
    level = _logger.info if status == "success" else _logger.warning
    level(f"TOOL   {kind} {tool_name:<30} {prefix}status={status} ({latency_str})")


def log_rag_retrieval(query: str, num_results: int, doc_type: str = None) -> None:
    """A RAG retrieval completing - what was searched for and how many
    results came back, visible live rather than only in a trace query."""
    filt = f" doc_type={doc_type}" if doc_type else ""
    _logger.info(f"RAG    retrieval query={query!r}{filt} -> {num_results} results")


def log_warning(source: str, message: str) -> None:
    """A non-fatal problem worth seeing immediately in the terminal —
    e.g. a Langfuse push failing. Used instead of Python's plain
    `logging` module for exactly this class of message so it's
    genuinely visible by default: a bare `logging.getLogger(...).warning()`
    call with no handler configured can print nothing at all depending
    on the app's overall logging setup, silently hiding a failure that
    someone actively debugging (e.g. "why isn't anything showing up in
    Langfuse") needs to actually see."""
    _logger.warning(f"{source.upper()}  {message}")
