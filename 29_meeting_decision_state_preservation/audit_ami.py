from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from decision_state import classify_state


def iter_json_files(root: Path):
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            yield path, None, f"{type(exc).__name__}: {exc}"
            continue
        yield path, obj, None


def _time(turn: dict, key: str):
    try:
        return float(turn[key])
    except (KeyError, TypeError, ValueError):
        return None


def audit(root: Path) -> dict:
    n_decisions = n_linked = n_multi = n_temporal = n_stateful = 0
    linked_turn_counts = Counter()
    state_counts = Counter()
    meetings = set()
    examples = []
    parse_errors = []
    duplicate_abstract_ids = 0
    malformed_items = 0
    ambiguous_state_windows = 0
    seen_abstract_ids = set()

    for path, obj, error in iter_json_files(root):
        if error:
            parse_errors.append({"file": str(path), "error": error})
            continue
        if not isinstance(obj, list):
            continue
        for item in obj:
            if not isinstance(item, dict):
                malformed_items += 1
                continue
            abstract = item.get("abstractive")
            extractive = item.get("extractive")
            if not isinstance(abstract, dict) or not isinstance(extractive, list):
                malformed_items += 1
                continue
            if str(abstract.get("type", "")).casefold() != "decisions":
                continue

            abstract_id = str(abstract.get("id", "")).strip()
            if abstract_id and abstract_id in seen_abstract_ids:
                duplicate_abstract_ids += 1
                continue
            if abstract_id:
                seen_abstract_ids.add(abstract_id)

            n_decisions += 1
            meetings.add(path.stem)
            linked = [turn for turn in extractive if isinstance(turn, dict) and str(turn.get("text", "")).strip()]
            linked.sort(key=lambda turn: (_time(turn, "starttime") is None, _time(turn, "starttime") or 0.0))
            n_linked += int(bool(linked))
            linked_turn_counts[len(linked)] += 1
            if len(linked) >= 2:
                n_multi += 1

            starts = [value for turn in linked if (value := _time(turn, "starttime")) is not None]
            ends = [value for turn in linked if (value := _time(turn, "endtime")) is not None]
            if starts and ends and max(ends) - min(starts) >= 15:
                n_temporal += 1

            joined = " ".join(str(turn["text"]) for turn in linked)
            parsed = classify_state(joined)
            state_counts[parsed.state] += 1
            n_stateful += int(parsed.explicit)
            ambiguous_state_windows += int(parsed.ambiguous)
            if len(examples) < 12 and linked:
                examples.append(
                    {
                        "decision": abstract.get("text"),
                        "n_links": len(linked),
                        "source_state": parsed.state,
                        "explicit_state_cue": parsed.explicit,
                        "ambiguous_state_window": parsed.ambiguous,
                        "source": [turn["text"] for turn in linked[:5]],
                    }
                )

    gates = {
        "G_support_200": n_decisions >= 200,
        "G_multiturn_100": n_multi >= 100,
        "G_temporal_75": n_temporal >= 75,
        "G_stateful_100": n_stateful >= 100,
    }
    return {
        "root": str(root),
        "n_decision_abstracts": n_decisions,
        "n_with_linked_utterances": n_linked,
        "n_multiturn_linked": n_multi,
        "n_temporally_extended_ge15s": n_temporal,
        "n_with_explicit_state_cue": n_stateful,
        "n_ambiguous_state_windows": ambiguous_state_windows,
        "linked_turn_hist": dict(sorted(linked_turn_counts.items())),
        "source_state_hist": dict(state_counts),
        "meeting_files_with_decisions": len(meetings),
        "duplicate_abstract_ids_skipped": duplicate_abstract_ids,
        "malformed_items": malformed_items,
        "json_parse_errors": parse_errors,
        "examples": examples,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ami_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
    print("decision abstracts", result["n_decision_abstracts"])
