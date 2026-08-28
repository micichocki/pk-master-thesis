"""Back-translation augmentation for rare emotion classes (NLLB-200).

    python gen_backtranslation.py                 # deu_Latn -> aug_bt_train.csv     (original)
    python gen_backtranslation.py --pivot ces_Latn  # -> aug_bt_cs_train.csv
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
RARE = ["strach", "zaufanie", "smutek"]
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()


@torch.no_grad()
def translate(texts: list[str], src: str, tgt: str, bs: int = 8) -> list[str]:
    """Length-sorted batched translation (fp16) to fit a 6 GB GPU."""
    tgt_id = tok.convert_tokens_to_ids(tgt)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[str | None] = [None] * len(texts)
    tok.src_lang = src
    for i in range(0, len(order), bs):
        chunk = order[i:i + bs]
        enc = tok([texts[j] for j in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=160, num_beams=1)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j, txt in zip(chunk, dec):
            out[j] = txt
    return out  # type: ignore[return-value]


def back_translate(texts: list[str], pivot: str) -> list[str]:
    mid = translate(texts, "pol_Latn", pivot)
    return translate(mid, pivot, "pol_Latn")


SECONDARY_PIVOTS = {
    "deu_Latn": ("bt_de", "aug_bt_train.csv"),      # original (frozen — do not overwrite lightly)
    "ces_Latn": ("bt_cs", "aug_bt_cs_train.csv"),   # Slavic pivot, closer to Polish
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pivot", default="deu_Latn", choices=sorted(SECONDARY_PIVOTS),
                    help="secondary pivot used for the rarest class ('strach')")
    args = ap.parse_args()
    tag2, out_name = SECONDARY_PIVOTS[args.pivot]

    df = pd.read_csv(PROCESSED / "twitteremo_train.csv")
    df["tekst"] = df["tekst"].fillna("")
    mask = df[RARE].sum(axis=1) > 0
    rare_df = df[mask].reset_index(drop=True)
    print(f"Rare-label rows to augment: {len(rare_df)}  (secondary pivot: {args.pivot})")

    aug_frames = []
    # EN pivot for all rare-label rows
    for pivot, tag, subset in [
        ("eng_Latn", "bt_en", rare_df),
        (args.pivot, tag2, rare_df[rare_df["strach"] == 1].reset_index(drop=True)),  # extra for rarest
    ]:
        t0 = time.time()
        print(f"  {tag}: {len(subset)} rows via {pivot} ...", flush=True)
        bt = back_translate(subset["tekst"].tolist(), pivot)
        a = subset.copy()
        a["tekst"] = bt
        a["clean_text"] = bt  # char-TF-IDF / HerBERT use raw 'tekst'; keep clean_text aligned
        a["aug_source"] = tag
        aug_frames.append(a)
        print(f"    done in {time.time() - t0:.0f}s")

    aug = pd.concat(aug_frames, ignore_index=True)
    # drop degenerate / empty paraphrases
    aug = aug[aug["tekst"].str.strip().str.len() > 0].reset_index(drop=True)
    out = PROCESSED / out_name
    aug.to_csv(out, index=False)
    print(f"Saved {out}  ({len(aug)} augmented rows)")
    print("Per-rare-class added:")
    for c in RARE:
        print(f"  {c}: +{int(aug[c].sum())}")


if __name__ == "__main__":
    main()
