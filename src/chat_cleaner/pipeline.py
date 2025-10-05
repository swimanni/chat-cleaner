import os
import json
import hashlib
import argparse
import pandas as pd
from typing import List, Dict, Generator

from .io_utils import discover_conversations
from .llamacpp_client import llm_structured_parse, get_model
from .preclean import preclean_text, light_preclean

# === Config ===
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MAX_WORKERS = 1   # ✅ single-threaded for stability
DEFAULT_MODEL = "models/mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf"


# === Normalizer ===
def normalize_for_llm(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", "\n").replace("\t", " ")
    text = text.replace("||", "\n").replace("｜｜", "\n")
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


# === Chunker (auto-tuning) ===
def chunkify(text: str, ctx_limit: int) -> Generator[str, None, None]:
    """Adjusts chunk size automatically based on model context window."""
    max_chars = int(ctx_limit * 0.35)  # 35% of context used for input
    lines = text.split("\n")
    chunk, length = [], 0
    for line in lines:
        L = len(line)
        if length + L > max_chars and chunk:
            yield "\n".join(chunk).strip()
            chunk, length = [line], L
        else:
            chunk.append(line)
            length += L
    if chunk:
        yield "\n".join(chunk).strip()


# === Cache ===
def cache_path(text: str, model: str) -> str:
    h = hashlib.md5((text + model).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


# === Core ===
def process_conversation(conv_id: str, raw_text: str, model: str, ctx_limit: int) -> List[Dict]:
    """Processes a single chat (Excel cell / PDF / text)."""
    if not raw_text or not str(raw_text).strip():
        return []

    try:
        cleaned = preclean_text(raw_text)
    except Exception:
        cleaned = light_preclean(raw_text)

    ai_ready = normalize_for_llm(cleaned)
    cpath = cache_path(ai_ready, model)
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)

    all_results: List[Dict] = []
    for chunk in chunkify(ai_ready, ctx_limit):
        part = llm_structured_parse(chunk, model)
        if part:
            all_results.extend(part)
        else:
            print(f"⚠️  Empty response for {conv_id} chunk ({len(chunk)} chars)")

    if all_results:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


def main(input_path: str, output_dir: str, model: str = DEFAULT_MODEL):
    os.makedirs(output_dir, exist_ok=True)
    chat_items = list(discover_conversations(input_path))
    print(f"🔍 Found {len(chat_items)} chat items to process.\n")

    # 🧠 Preload model once globally
    print(f"🧠 Preloading model for all rows: {model}")
    llm = get_model(model)
    ctx_limit = llm.n_ctx()
    print(f"⚙️  Using context window: {ctx_limit} tokens")

    # === Sequential processing (1 worker) ===
    for cid, text in chat_items:
        try:
            result = process_conversation(cid, text, model, ctx_limit)
            if not result:
                print(f"⚠️  No data returned for {cid}")
                continue
            df = pd.DataFrame(result)
            out_path = os.path.join(output_dir, f"{cid}_clean.csv")
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Processed: {cid} → {os.path.basename(out_path)}")
        except Exception as e:
            msg = str(e).encode("utf-8", errors="ignore").decode()
            print(f"Error processing {cid}: {msg}")

    print("\n🎯 All chats processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI chat cleaner (single-threaded, auto-tuned)")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output directory for CSVs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    args = parser.parse_args()
    main(args.input, args.output, args.model)
