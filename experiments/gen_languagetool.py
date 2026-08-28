"""Correct TwitterEmo text with LanguageTool (PL), parallelised — for correction tests #1/#2."""
from __future__ import annotations
import time
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = PROCESSED / "lt"; OUT.mkdir(exist_ok=True)
N_WORKERS = 4

_tool = None
def _init():
    global _tool
    import language_tool_python as ltp
    _tool = ltp.LanguageTool("pl-PL")

def _correct(s: str) -> str:
    s = s if isinstance(s, str) else ""
    if not s.strip():
        return s
    try:
        return _tool.correct(s)
    except Exception:
        return s

def main() -> None:
    with Pool(N_WORKERS, initializer=_init) as pool:
        for split in ["train", "val", "test"]:
            dst = OUT / f"twitteremo_{split}.csv"
            if dst.exists():
                print(f"skip {dst.name}"); continue
            df = pd.read_csv(PROCESSED / f"twitteremo_{split}.csv")
            texts = df["tekst"].fillna("").tolist()
            print(f"=== twitteremo_{split}: {len(texts)} tekstów ({N_WORKERS} workerów) ===", flush=True)
            t0 = time.time()
            corrected = pool.map(_correct, texts, chunksize=100)
            pd.DataFrame({"text_lt": corrected}).to_csv(dst, index=False)
            print(f"    saved {dst.name} in {time.time()-t0:.0f}s ({len(texts)/(time.time()-t0):.0f}/s)", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
