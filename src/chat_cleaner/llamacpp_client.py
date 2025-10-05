import os
import re
import json
from json_repair import repair_json
from llama_cpp import Llama
from .prompts import SYSTEM_PROMPT, USER_INSTRUCTION_TEMPLATE

# ------------------------------------------------------
# ⚙️ Model cache to reuse loaded models (saves time)
# ------------------------------------------------------
_MODELS = {}


def get_model(model_path: str) -> Llama:
    """Load and cache the llama.cpp model once."""
    abs_path = os.path.abspath(model_path)
    if abs_path not in _MODELS:
        print(f"🧠 Loading model from {abs_path}")
        _MODELS[abs_path] = Llama(
            model_path=abs_path,
            n_ctx=32768,        # large context
            n_threads=6,        # tune to your CPU cores
            n_gpu_layers=0,     # CPU-only
            verbose=False,
        )
    return _MODELS[abs_path]


# ------------------------------------------------------
# 🧰 Robust JSON repair helper
# ------------------------------------------------------
def try_repair_json(bad_json: str) -> str:
    """
    Two-stage JSON repair:
      1. Light regex cleanup (escapes, commas, brackets)
      2. Deep fix with json-repair
    Returns the repaired JSON string.
    """
    if not bad_json or not isinstance(bad_json, str):
        return bad_json

    text = bad_json.strip()

    # Stage 1️⃣ Light regex cleanup
    text = re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)  # invalid escapes
    text = re.sub(r',\s*(?=[}\]])', '', text)         # trailing commas
    text = re.sub(r',\s*,', ',', text)                # double commas
    text = re.sub(r'[\x00-\x1f]', '', text)           # control chars

    # Balance braces/brackets
    open_braces, close_braces = text.count('{'), text.count('}')
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        text = '{' * (close_braces - open_braces) + text

    open_brackets, close_brackets = text.count('['), text.count(']')
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        text = '[' * (close_brackets - open_brackets) + text

    # Stage 2️⃣ Deep repair with json-repair
    try:
        repaired = repair_json(text)
        return repaired
    except Exception:
        return text


# ------------------------------------------------------
# 🧩 JSON grammar for llama.cpp (if supported)
# ------------------------------------------------------
def _json_grammar() -> str:
    return r'''
root            ::= ws array ws
array           ::= "[" ws (object (ws "," ws object)*)? ws "]"
object          ::= "{" ws time_kv ws "," ws speaker_kv ws "," ws role_kv ws "," ws message_kv ws "}"
time_kv         ::= "\"time\"" ws ":" ws (string | "null")
speaker_kv      ::= "\"speaker\"" ws ":" ws string
role_kv         ::= "\"role\"" ws ":" ws ("\"Agent\"" | "\"User\"" | "\"System\"")
message_kv      ::= "\"message\"" ws ":" ws string
string          ::= "\"" chars "\""
chars           ::= char*
char            ::= [^"\\] | escape
escape          ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex             ::= [0-9a-fA-F]
ws              ::= ([ \t\n\r])*
'''


# ------------------------------------------------------
# 🧠 Core parse function
# ------------------------------------------------------
def llm_structured_parse(raw_text: str, model: str):
    """
    Sends conversation text to the model and returns a clean JSON array
    of {time, speaker, role, message} objects.
    Handles malformed JSON via automatic repair.
    """

    llm = get_model(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(raw_text=raw_text).strip()},
    ]

    grammar = _json_grammar()
    grammar_supported = True

    def call(max_tokens: int):
        """Wrapper for llama_cpp call with grammar fallback."""
        nonlocal grammar_supported
        try:
            result = llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=1.0,
                stop=[],
            )
        except Exception as e:
            if "'_grammar'" in str(e) or "unsupported" in str(e).lower():
                print("⚠️  Grammar mode unsupported, retrying without grammar.")
                grammar_supported = False
                result = llm.create_chat_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    stop=[],
                )
            else:
                raise e
        return result["choices"][0]["message"]["content"]

    # --- first attempt ---
    content = call(2048)
    json_text = _extract_json(content)

    # --- fallback with more tokens if short ---
    if len(json_text) < 3:
        content = call(3072)
        json_text = _extract_json(content)

    if not json_text.strip():
        print("⚠️  No JSON detected in model output.")
        return []

    # --- parse + repair if necessary ---
    try:
        return json.loads(json_text)
    except Exception as e:
        print(f"⚠️  JSON load failed: {e}")
        repaired = try_repair_json(json_text)
        try:
            return json.loads(repaired)
        except Exception as e2:
            print(f"❌ Still invalid after repair: {e2}\nSnippet:\n{repaired[:600]}")
            return []


# ------------------------------------------------------
# 🧩 Extract JSON safely
# ------------------------------------------------------
def _extract_json(content: str) -> str:
    start, end = content.find("["), content.rfind("]") + 1
    if start == -1 or end <= start:
        return ""
    return content[start:end]
