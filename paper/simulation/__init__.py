from importlib import import_module
from typing import Any

from paper.simulation.core import (
    MODE_DESCRIPTIONS,
    EVENT_ATTRIBUTION_MODES,
    SimulationConfig as EventAttributionSimulationConfig,
    simulate_event_attribution_dataset,
)

_LAZY_EXPORTS = {
    "EXPERIMENT_IDS": ("paper.simulation.differential_pipe", "EXPERIMENT_IDS"),
    "SCENARIOS": ("paper.simulation.differential_pipe", "SCENARIOS"),
    "DifferentialSimulationConfig": (
        "paper.simulation.differential_pipe",
        "DifferentialSimulationConfig",
    ),
    "DifferentialSimulationPipe": (
        "paper.simulation.differential_pipe",
        "DifferentialSimulationPipe",
    ),
    "ExperimentSpec": ("paper.simulation.differential_pipe", "ExperimentSpec"),
    "experiment_specs": ("paper.simulation.differential_pipe", "experiment_specs"),
    "get_experiment_spec": ("paper.simulation.differential_pipe", "get_experiment_spec"),
    "simulate_differential_experiment": (
        "paper.simulation.differential_pipe",
        "simulate_differential_experiment",
    ),
    "SCENARIO_NAMES": ("paper.simulation.direction_pipe", "SCENARIO_NAMES"),
    "SimPipe": ("paper.simulation.direction_pipe", "SimPipe"),
    "SimulationConfig": ("paper.simulation.direction_pipe", "SimulationConfig"),
    "simulation": ("paper.simulation.direction_pipe", "simulation"),
    "EventAttributionSimulationPipe": (
        "paper.simulation.event_attribution_pipe",
        "EventAttributionSimulationPipe",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "MODE_DESCRIPTIONS",
    "EVENT_ATTRIBUTION_MODES",
    "EventAttributionSimulationConfig",
    "EventAttributionSimulationPipe",
    "simulate_event_attribution_dataset",
    "EXPERIMENT_IDS",
    "SCENARIOS",
    "DifferentialSimulationConfig",
    "DifferentialSimulationPipe",
    "ExperimentSpec",
    "experiment_specs",
    "get_experiment_spec",
    "simulate_differential_experiment",
    "SimPipe",
    "SCENARIO_NAMES",
    "SimulationConfig",
    "simulation",
]
