"""Pipeline and CLI for event-attribution simulations."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.simulation.core import (
    EVENT_ATTRIBUTION_MODES,
    SimulationConfig,
    save_h5ad,
    simulate_event_attribution_dataset,
)

DEFAULT_OUT = ROOT / "paper" / "data" / "event_attribution" / "raw"
logger = logging.getLogger(__name__)

class EventAttributionSimulationPipe:
    def __init__(self, output_dir: str | Path = DEFAULT_OUT, config: SimulationConfig = SimulationConfig()) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def run(self, modes: Iterable[str] | None = None) -> list[Path]:
        selected = list(EVENT_ATTRIBUTION_MODES if modes is None else modes)
        unknown = [mode for mode in selected if mode not in EVENT_ATTRIBUTION_MODES]
        if unknown:
            raise ValueError(f"Unknown modes {unknown}; expected {EVENT_ATTRIBUTION_MODES}")
        outputs: list[Path] = []
        for mode in selected:
            logger.info("Generating event-attribution simulation: %s", mode)
            adata = simulate_event_attribution_dataset(mode, self.config)
            outputs.append(save_h5ad(adata, self.output_dir / f"{mode}.h5ad"))
        return outputs


def build_arg_parser(default_output_dir: Path = DEFAULT_OUT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--mode", action="append", choices=EVENT_ATTRIBUTION_MODES, dest="modes")
    parser.add_argument("--n-cells", type=int, default=SimulationConfig.n_cells)
    parser.add_argument("--n-genes", type=int, default=SimulationConfig.n_genes)
    parser.add_argument("--n-modules", type=int, default=SimulationConfig.n_modules)
    parser.add_argument("--effect-strength", type=float, default=SimulationConfig.expression_effect_strength)
    parser.add_argument("--spliced-dropout", type=float, default=SimulationConfig.spliced_dropout)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_modules=args.n_modules,
        expression_effect_strength=args.effect_strength,
        spliced_dropout=args.spliced_dropout,
        seed=args.seed,
    )


def main(default_output_dir: Path = DEFAULT_OUT) -> None:
    args = build_arg_parser(default_output_dir).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    EventAttributionSimulationPipe(args.output_dir, config_from_args(args)).run(args.modes)


if __name__ == "__main__":
    main()
