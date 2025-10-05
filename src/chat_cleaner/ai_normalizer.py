import html
import re

def normalize_for_llm(text: str) -> str:
    """
    Minimal AI-friendly normalizer that helps the model separate speakers and messages
    without regex-heavy logic. Keeps punctuation, spacing, and emojis intact.
    """

    if not isinstance(text, str):
        text = str(text)

    # 1️⃣ Decode HTML and unify newlines
    text = html.unescape(text)
    text = text.replace("\r", "\n")
    text = text.replace("||", "\n")
    text = text.replace("｜｜", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # 2️⃣ Add soft cues for speaker/message breaks
    #    These help LLM recognize "Name:" or "Name-" as new messages
    text = re.sub(r"([A-Za-z0-9)\]])(:|-)\s+", r"\1\2\n", text)
    # Example: "Ravi: ok" → "Ravi:\n ok" ; "neha- today" → "neha-\n today"

    # 3️⃣ Encourage split after complete sentences if next token looks like a name
    text = re.sub(r"([.?!])\s+([A-Z][a-z]+[-:])", r"\1\n\2", text)
    # Example: "since when? Neha-" → "since when?\nNeha-"

    # 4️⃣ Collapse excessive newlines but keep boundaries
    text = re.sub(r"\n{2,}", "\n", text).strip()

    # 5️⃣ Chunk gently if text is too large
    if len(text) > 1500:
        lines = text.split("\n")
        chunks, chunk = [], []
        length = 0
        for line in lines:
            length += len(line)
            chunk.append(line)
            if length > 700:
                chunks.append("\n".join(chunk))
                chunk, length = [], 0
        if chunk:
            chunks.append("\n".join(chunk))
        text = "\n\n---\n\n".join(chunks)

    return text.strip()
