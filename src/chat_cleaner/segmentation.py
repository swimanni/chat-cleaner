import os
import json
from .llamacpp_client import get_model

def segment_conversations_ai(text: str, model_path: str, cache_dir: str = "cache"):
    """
    Uses the AI model to detect and split multiple chat sessions inside a raw text file.
    Returns a list of conversation text chunks.
    """

    # --- Caching for speed ---
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = str(abs(hash(text)))[:16]
    cache_file = os.path.join(cache_dir, f"seg_{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # --- Build the prompt ---
    prompt = f"""
You are a conversation separator AI. The input text may contain multiple customer support chats, one after another.

Task:
- Split the text into distinct conversations.
- Detect boundaries when the context clearly resets (e.g. new greeting, new timestamp, new customer name, "Internal Participant(s):" etc.)
- Return only a JSON array of strings, each string being one full conversation.

Example output:
[
  "Conversation 1 text ...",
  "Conversation 2 text ..."
]

Do not summarize, do not label, just cleanly separate them.

TEXT:
{text}
"""

    try:
        llm = get_model(model_path)
        result = llm.create_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=4096,
        )
        content = result["choices"][0]["text"].strip()

        # --- extract possible JSON ---
        start, end = content.find("["), content.rfind("]") + 1
        snippet = content[start:end] if start != -1 and end > start else content

        try:
            data = json.loads(snippet)
            if isinstance(data, list) and all(isinstance(c, str) for c in data):
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return data
        except Exception:
            pass

        # fallback: treat as single conversation
        return [text.strip()]

    except Exception as e:
        print(f"⚠️ Segmentation failed: {e}")
        return [text.strip()]
