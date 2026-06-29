"""Materialize the Django-RCX SFT mix to a local jsonl, byte-identical to the
selection `lora_django_trial.py`'s `DjangoRcxBuilder` feeds Tinker.

Mirrors that builder exactly: load each `data/<unit>/django.jsonl` from
`swespot/sft-v0`, shuffle(seed=0) and cap at 2048 rows per unit, concatenate the
four units in order. Result: 8039 rows of {"messages": [...]}, which torchtune's
`chat_dataset(source="json", conversation_style="openai")` consumes directly.

Run:  /home/alex/SkyRL/.venv/bin/python train/prepare_django_rcx_data.py
"""

import json
import os

import datasets

DJANGO_RCX_UNITS = ["software_design", "ctx_impl", "evo_replay", "rt_alignment"]
HF_DATASET = "swespot/sft-v0"
PER_UNIT_CAP = 2048
CAP_SEED = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "django_rcx_8039.jsonl")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    parts = []
    counts = {}
    for i, unit in enumerate(DJANGO_RCX_UNITS, 1):
        unit_ds = datasets.load_dataset(
            HF_DATASET, data_files=f"data/{unit}/django.jsonl", split="train"
        )
        if PER_UNIT_CAP and len(unit_ds) > PER_UNIT_CAP:
            unit_ds = unit_ds.shuffle(seed=CAP_SEED).select(range(PER_UNIT_CAP))
        counts[unit] = len(unit_ds)
        print(f"  [{i}/{len(DJANGO_RCX_UNITS)}] {unit}: {len(unit_ds)} rows", flush=True)
        parts.append(unit_ds)

    ds = datasets.concatenate_datasets(parts)
    n = 0
    with open(OUT, "w") as f:
        for row in ds:
            # keep only the messages field, in OpenAI {"messages":[...]} form
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
            n += 1

    print(f"\nper-unit counts : {counts}")
    print(f"total rows      : {n}")
    print(f"written to      : {OUT}")
    assert n == sum(counts.values()), "row count mismatch"
    # the expected fingerprint from lora_django_trial.py's docstring
    expected = {"software_design": 2048, "ctx_impl": 1895, "evo_replay": 2048, "rt_alignment": 2048}
    if counts == expected:
        print("PARITY OK: per-unit counts match lora_django_trial.py (8039 total)")
    else:
        print(f"WARNING: counts differ from expected {expected}")


if __name__ == "__main__":
    main()
