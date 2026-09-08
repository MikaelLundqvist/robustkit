"""
Shared helpers for the openml_suite validation scripts.

Kept deliberately minimal and dependency-free (beyond what's already
used everywhere else) so each run_*.py module can import from here
without pulling in unrelated machinery.
"""


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_run(label, fn):
    """
    Run fn(), reporting and swallowing any exception rather than
    aborting the whole validation run. Returns the result on success,
    None on failure (after printing the failure).
    """
    try:
        result = fn()
        print(f"  [OK]   {label}")
        return result
    except Exception as exc:
        print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        return None
