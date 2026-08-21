from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class FamilyRenderer:
    key: str
    title: str
    description: str
    action_a: str
    action_b: str
    wait: str

    @property
    def surface_actions(self) -> tuple[str, str, str]:
        return (self.action_a, self.action_b, self.wait)

    def action_map(self) -> dict[str, str]:
        return {
            self.action_a: "a",
            self.action_b: "b",
            self.wait: "wait",
        }

    def render_start(self, episode: int, instance: int) -> str:
        return (
            f"New task — {self.title}, unit {instance}. {self.description} "
            f"Available actions: {self.action_a}, {self.action_b}, {self.wait}."
        )

    def render_trial(self, trial: int) -> str:
        return (
            f"Round {trial}. Choose one action: "
            f"{self.action_a}, {self.action_b}, or {self.wait}."
        )


_FAMILIES = [
    FamilyRenderer("greenhouse", "greenhouse control", "Keep the plant-health indicator in its acceptable zone.", "adjust_irrigation", "adjust_lighting", "hold_settings"),
    FamilyRenderer("safe", "safe recovery", "Restore the locking mechanism to a working state.", "turn_left_dial", "turn_right_dial", "leave_dials"),
    FamilyRenderer("machine", "machine calibration", "Bring the production-quality indicator into range.", "tune_pressure", "tune_cooling", "keep_machine"),
    FamilyRenderer("store", "store operation", "Improve the customer-response indicator for this period.", "change_price", "change_display", "keep_plan"),
    FamilyRenderer("logistics", "delivery routing", "Bring the on-time-delivery indicator into range.", "reroute_trucks", "reassign_stock", "keep_schedule"),
    FamilyRenderer("water", "water treatment", "Restore the water-quality indicator to the target zone.", "adjust_filter", "adjust_flow", "hold_controls"),
    FamilyRenderer("server", "server recovery", "Return the service-health indicator to normal.", "shift_traffic", "restart_worker", "leave_service"),
    FamilyRenderer("museum", "museum climate", "Keep the preservation-risk indicator within its safe zone.", "adjust_humidity", "adjust_airflow", "hold_climate"),
    FamilyRenderer("farm", "crop management", "Improve the crop-condition indicator for this plot.", "change_nutrients", "change_shading", "keep_treatment"),
    FamilyRenderer("battery", "battery system", "Return the battery-health indicator to its target range.", "change_charge_rate", "change_cooling", "hold_battery"),
    FamilyRenderer("traffic", "traffic control", "Bring the congestion indicator into the acceptable range.", "change_signal_timing", "open_bypass", "keep_traffic_plan"),
    FamilyRenderer("lab", "lab process", "Restore the assay-quality indicator to its target range.", "change_reagent_feed", "change_temperature", "hold_process"),
    FamilyRenderer("orbital_station", "orbital station stabilization", "Bring the stability indicator into its safe operating range.", "adjust_gyroscope", "adjust_thruster", "hold_course"),
]

FAMILIES: Dict[str, FamilyRenderer] = {f.key: f for f in _FAMILIES}
TRAIN_FAMILIES = tuple(f.key for f in _FAMILIES if f.key != "orbital_station")


def get_family(key: str) -> FamilyRenderer:
    try:
        return FAMILIES[key]
    except KeyError as exc:
        raise KeyError(f"unknown family {key!r}") from exc


def schedule_for(diversity: str, episodes: int, pair_id: int = 0) -> list[str]:
    if episodes > len(TRAIN_FAMILIES):
        raise ValueError("not enough distinct training families")
    pool = list(TRAIN_FAMILIES[:episodes])
    offset = pair_id % len(pool)
    if diversity == "concentrated":
        return [pool[offset]] * episodes
    if diversity == "distributed":
        return pool[offset:] + pool[:offset]
    raise ValueError(f"unknown diversity condition: {diversity}")


def all_surface_actions(keys: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for key in keys:
        out.update(get_family(key).surface_actions)
    return out
