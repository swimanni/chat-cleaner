from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd
from pypdf import PdfReader


def discover_conversations(input_path: str) -> Iterable[Tuple[str, str]]:
    """
    Yields (conversation_id, raw_text) pairs for the given input.
    - For .xlsx/.xls/.csv: each row becomes a separate conversation.
      Each non-empty cell is joined with a newline so multi-cell text
      doesn't collapse into one line.
    - For .txt: optionally segmented by AI if available.
    - For .pdf: all pages joined together with newlines.
    """
    p = Path(input_path)
    ext = p.suffix.lower()

    # =====================
    # Excel / CSV handling
    # =====================
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(p, dtype=str, keep_default_na=False)
        for i, row in df.iterrows():
            cells = [c.strip() for c in row.tolist() if isinstance(c, str) and c.strip()]
            # Join cells with newlines to preserve separation
            raw = "\n".join(cells).strip()
            if raw:
                yield (f"{p.stem}_row{i+1}", raw)

    elif ext == ".csv":
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        for i, row in df.iterrows():
            cells = [c.strip() for c in row.tolist() if isinstance(c, str) and c.strip()]
            raw = "\n".join(cells).strip()
            if raw:
                yield (f"{p.stem}_row{i+1}", raw)

    # =====================
    # Text file handling
    # =====================
    elif ext == ".txt":
        raw = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return

        # Try AI-based segmentation (optional)
        try:
            from .segmentation import segment_conversations_ai
            chats = segment_conversations_ai(raw, "models/mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf")
        except Exception as e:
            print(f"⚠️  AI segmentation failed ({e}), using fallback.")
            chats = [raw]

        for i, chunk in enumerate(chats, start=1):
            yield (f"{p.stem}_row{i}", chunk)

    # =====================
    # PDF file handling
    # =====================
    elif ext == ".pdf":
        reader = PdfReader(str(p))
        pages = []
        for pg in reader.pages:
            t = pg.extract_text() or ""
            if t.strip():
                pages.append(t.strip())
        raw = "\n".join(pages).strip()
        if raw:
            yield (p.stem, raw)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
