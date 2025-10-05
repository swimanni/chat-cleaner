import os
import re
import json
import requests
from typing import List, Dict
from .prompts import SYSTEM_PROMPT, USER_INSTRUCTION_TEMPLATE

# === Ollama settings ===
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")

# === Prompt builders ===
def _build_messages(raw_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(raw_text=raw_text)},
    ]

# === Streaming helper ===
def _stream_response(resp) -> str:
    """Read Ollama stream incrementally and collect generated text."""
    full = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("message", {}).get("content", "")
            full += chunk
        except Exception:
            continue
    return full.strip()

# === JSON repair ===
def safe_json_load(text: str):
    start, end = text.find("["), text.rfind("]") + 1
    snippet = text[start:end] if start != -1 and end > start else text
    snippet = (
        snippet.replace("\n", " ")
               .replace("'", '"')
               .replace('}{', '},{')
               .replace('",,', '",')
               .replace('], [', '],')
    )
    snippet = snippet.replace('"}{"', '"},{"')

    # close open quotes/braces if model stopped mid-string
    if snippet.count('"') % 2 != 0:
        snippet += '"'
    if snippet.count("{") > snippet.count("}"):
        snippet += "}" * (snippet.count("{") - snippet.count("}"))
    if not snippet.strip().endswith("]"):
        snippet += "]"

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        snippet = re.sub(r",\s*]", "]", snippet)
        return json.loads(snippet)


# === Main LLM call ===
def llm_structured_parse(raw_text: str, model: str = None) -> List[Dict]:
    """
    Send raw_text to Ollama Chat API and return parsed JSON (list of dicts).
    Repairs minor JSON issues and retries with longer output if still malformed.
    """
    model = model or DEFAULT_MODEL

    def call_ollama(num_predict=1800):
        payload = {
            "model": model,
            "messages": _build_messages(raw_text),
            "stream": True,
            "options": {
                "num_predict": 2000,      # allow longer outputs
                "temperature": 0.1,       # less randomness
                "stop": ["```", "</s>"],  # stop before junk
                "top_p": 0.9,
            },
        }
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=240) as resp:
            resp.raise_for_status()
            return _stream_response(resp).strip()

    # --- first call ---
    content = call_ollama(300)
    if not content.strip().endswith("]"):
        content = content.rstrip(",") + "]"

    try:
        return safe_json_load(content)
    except Exception as e1:
        print(f"⚠️  Parse failed ({e1}); retrying with longer generation ...")
        content2 = call_ollama(1000)
        if not content2.strip().endswith("]"):
            content2 = content2.rstrip(",") + "]"
        try:
            return safe_json_load(content2)
        except Exception as e2:
            print(f"❌  Final parse failed: {e2}")
            repaired = content2.replace('“', '"').replace('”', '"')
            if not repaired.strip().endswith("]"):
                repaired += "]"
            try:
                return safe_json_load(repaired)
            except Exception:
                print("🔎 Raw snippet (first 500 chars):")
                print(repaired[:500])
                raise
