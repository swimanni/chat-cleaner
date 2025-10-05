import re

def preclean_text(text: str) -> str:
    """
    Light-weight cleaner that removes obvious noise and structure symbols
    but preserves chat semantics and emojis.
    Adds soft newline hints when multiple people talk in one line.
    """

    # Normalize spaces and line breaks
    text = re.sub(r"\s+", " ", text)

    # Remove boilerplate metadata
    text = re.sub(r"(Conversation ID|Session ID|Chat Transcript|Internal Participant|Bot/Flow).*?:.*?(?=\s[A-Z]|$)", "", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"(?:http|https)://\S+", "", text)

    # Remove obvious separators
    text = re.sub(r"[<>\[\]\|\\]", " ", text)
    text = re.sub(r"[-=_]{3,}", " ", text)

    # --- Key Fix ---
    # Insert a newline if one person finishes and another starts mid-line
    # like: "ok. since when? neha- today only"
    text = re.sub(
        r"([.!?])\s*([A-Z]?[a-z]+[-–—])",
        r"\1\n\2",
        text
    )

    # Optional: ensure newline before known speaker cues
    text = re.sub(r"\s+(Agent|User|Customer|Client|Rep|Ravi|Neha|Tani|Jose)\s*[:\-]", r"\n\1:", text, flags=re.IGNORECASE)

    return text.strip()


def light_preclean(text: str):
    """Minimal fallback when full preclean fails."""
    return text.replace("\r", " ").replace("\n", " ").strip()
