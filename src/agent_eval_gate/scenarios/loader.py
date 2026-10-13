"""Load scenarios from YAML and validate them against the schema.

YAML rather than JSON because 50+ of these are written by hand: it allows
comments explaining why a trap is a trap, and folded blocks for task text.

Validation happens at load time, not at scoring time. A typo in one file
should fail immediately, naming the file and the field, rather than surfacing
as a confusing failure in the middle of a paid evaluation run.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval_gate.scenarios.schema import Scenario

EXAMPLES_DIR = Path(__file__).parent / "examples"


class ScenarioLoadError(Exception):
    """A scenario file is missing, unparseable, or fails validation."""


def load_scenario(path: Path) -> Scenario:
    """Load and validate a single scenario file."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ScenarioLoadError(f"{path.name}: not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioLoadError(f"{path.name}: expected a mapping, got {type(raw).__name__}")

    try:
        return Scenario(**raw)
    except ValidationError as exc:
        raise ScenarioLoadError(f"{path.name}: {exc}") from exc


def load_scenarios(directory: Path | None = None) -> list[Scenario]:
    """Load every .yaml scenario in a directory, sorted by id.

    Sorted so that a run is reproducible: the same suite in the same order on
    every machine, independent of filesystem listing order.
    """
    directory = directory or EXAMPLES_DIR
    scenarios = [load_scenario(path) for path in sorted(directory.glob("*.yaml"))]

    ids = [scenario.id for scenario in scenarios]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ScenarioLoadError(f"duplicate scenario ids: {sorted(duplicates)}")

    return sorted(scenarios, key=lambda s: s.id)
