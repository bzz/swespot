"""Materialize the Django-RCX SFT mix to a local jsonl, byte-identical to the
selection `lora_django_trial.py`'s `DjangoRcxBuilder` feeds Tinker.

Mirrors that builder exactly: load each `data/<unit>/django.jsonl` from
`swespot/sft-v0`, shuffle(seed=0) and cap at 2048 rows per unit, concatenate the
four units in order. Result: 8039 rows of {"messages": [...]}, which torchtune's
`chat_dataset(source="json", conversation_style="openai")` consumes directly.

Any rows beyond the per-unit cap (the shuffled *tail*, never seen in training)
are written to a separate `django_rcx_val.jsonl` to use as a held-out validation
set. Because both splits come from the same seed-0 shuffle, train and val are a
clean disjoint partition of each unit. (Units at or below the cap contribute no
val rows; with the current data only ctx_impl is below cap.)

Run:  /home/alex/SkyRL/.venv/bin/python train/prepare_django_rcx_data.py
"""

import json
import os

import datasets

DJANGO_RCX_UNITS = ["software_design", "ctx_impl", "evo_replay", "rt_alignment"]
HF_DATASET = "swespot/sft-v0"
PER_UNIT_CAP = 2048
CAP_SEED = 0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "torchtune")
OUT = os.path.join(DATA_DIR, "django_rcx_8039.jsonl")
OUT_VAL = os.path.join(DATA_DIR, "django_rcx_val.jsonl")


def _write_messages(path: str, rows) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            # keep only the messages field, in OpenAI {"messages":[...]} form
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    train_parts = []
    val_parts = []
    counts = {}
    val_counts = {}
    for i, unit in enumerate(DJANGO_RCX_UNITS, 1):
        unit_ds = datasets.load_dataset(
            HF_DATASET, data_files=f"data/{unit}/django.jsonl", split="train"
        )
        if PER_UNIT_CAP and len(unit_ds) > PER_UNIT_CAP:
            shuffled = unit_ds.shuffle(seed=CAP_SEED)
            train_ds = shuffled.select(range(PER_UNIT_CAP))
            val_ds = shuffled.select(range(PER_UNIT_CAP, len(shuffled)))
        else:
            train_ds = unit_ds
            val_ds = unit_ds.select(range(0))  # empty, same schema
        counts[unit] = len(train_ds)
        val_counts[unit] = len(val_ds)
        print(
            f"  [{i}/{len(DJANGO_RCX_UNITS)}] {unit}: {len(train_ds)} train / "
            f"{len(val_ds)} val rows",
            flush=True,
        )
        train_parts.append(train_ds)
        if len(val_ds):
            val_parts.append(val_ds)

    train = datasets.concatenate_datasets(train_parts)
    n = _write_messages(OUT, train)

    n_val = 0
    if val_parts:
        val = datasets.concatenate_datasets(val_parts)
        n_val = _write_messages(OUT_VAL, val)

    print(f"\nper-unit train counts : {counts}")
    print(f"per-unit val counts   : {val_counts}")
    print(f"total train rows      : {n}")
    print(f"total val rows        : {n_val}")
    print(f"train written to      : {OUT}")
    print(f"val written to        : {OUT_VAL}")
    assert n == sum(counts.values()), "train row count mismatch"
    assert n_val == sum(val_counts.values()), "val row count mismatch"
    # the expected fingerprint from lora_django_trial.py's docstring
    expected = {"software_design": 2048, "ctx_impl": 1895, "evo_replay": 2048, "rt_alignment": 2048}
    if counts == expected:
        print("PARITY OK: per-unit train counts match lora_django_trial.py (8039 total)")
    else:
        print(f"WARNING: train counts differ from expected {expected}")


if __name__ == "__main__":
    main()
