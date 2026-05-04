"""Smoke tests covering all (case, da_method) combinations of scripts/main.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = REPO_ROOT / "scripts" / "main.py"

CASES = ["lorenz_63", "lorenz_96", "kuramoto", "coupled_kuramoto"]
DA_METHODS = ["enkf", "agmf", "pff", "particle_filter"]

# Per-case extras layered on top of the shared smoke-test overrides.
# coupled_kuramoto's defaults (state_dim=1024, spinup_steps=25) are too heavy
# for a smoke test, so shrink the spatial grid and skip spinup here.
CASE_EXTRA_OVERRIDES: dict[str, list[str]] = {
    "coupled_kuramoto": [
        "case.state_dim=64",
        "case.domain_length=64",
        "spinup_steps=0",
    ],
}


# (case, da_method) combos that don't currently run end-to-end.
# pff's kernel assumes a flat state of size `state_dim`, so it breaks for cases
# with `num_states > 1` (e.g. coupled_kuramoto). Tracked separately from this
# smoke test.
KNOWN_FAILURES: set[tuple[str, str]] = {
    ("coupled_kuramoto", "pff"),
}


@pytest.mark.parametrize("case", CASES)  # type: ignore[misc]
@pytest.mark.parametrize("da_method", DA_METHODS)  # type: ignore[misc]
def test_main_combination(case: str, da_method: str) -> None:
    """Run scripts/main.py end-to-end for every (case, da_method) pair."""
    if (case, da_method) in KNOWN_FAILURES:
        pytest.xfail(f"{case} + {da_method} is a known incompatibility")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [
            sys.executable,
            str(MAIN_SCRIPT),
            f"case={case}",
            f"da_method={da_method}",
            "data_assimilation_steps=10",
            "model_integration_steps=5",
            "ensemble_size=50",
            *CASE_EXTRA_OVERRIDES.get(case, []),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"case={case} da_method={da_method} failed (exit={result.returncode})\n"
        f"--- stdout ---\n{result.stdout.decode(errors='replace')}\n"
        f"--- stderr ---\n{result.stderr.decode(errors='replace')}"
    )
