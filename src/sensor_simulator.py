"""SIMULATED DATA generator. Not connected to real industrial hardware."""
import csv
import time

import numpy as np

from src.config import STREAMING_DIR

SCENARIOS = ["Normal", "High Tool Wear", "Overheating", "Power Overload", "Random Spike"]
_rng = np.random.default_rng()


def generate_reading(scenario: str = "Normal") -> dict:
    air = _rng.normal(300, 1.5)
    r = {"air_temp": air, "process_temp": air + 10 + _rng.normal(0, 0.8),
         "rpm": _rng.normal(1500, 80), "torque": _rng.normal(40, 5), "tool_wear": _rng.uniform(10, 150)}
    if scenario == "High Tool Wear":
        r.update(tool_wear=_rng.uniform(215, 250), torque=_rng.normal(58, 3))
    elif scenario == "Overheating":
        r.update(process_temp=air + _rng.uniform(7.0, 8.2), rpm=_rng.normal(1300, 30), torque=_rng.normal(52, 3))
    elif scenario == "Power Overload":
        r.update(rpm=_rng.normal(1250, 30), torque=_rng.normal(72, 3))
    elif scenario == "Random Spike":
        r.update(rpm=_rng.uniform(2500, 2850), torque=_rng.uniform(4, 10), air_temp=air + 3)
    return {k: round(float(v), 2) for k, v in r.items()}


def stream(scenario: str = "Normal", interval: float = 1.0, count: int = 10):
    """Yield readings at a configurable interval (used by CLI demo)."""
    for _ in range(count):
        yield generate_reading(scenario)
        time.sleep(interval)


def save_to_csv(reading: dict, scenario: str):
    path = STREAMING_DIR / "simulated_stream.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "scenario", *reading])
        if new:
            w.writeheader()
        w.writerow({"label": "SIMULATED DATA", "scenario": scenario, **reading})


if __name__ == "__main__":
    from src.pipeline import process
    for rd in stream("Normal", 1.0, 5):
        print("SIMULATED DATA", process(rd, source="SIMULATED"))
