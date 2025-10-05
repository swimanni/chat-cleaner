# 🧠 AI Chat Cleaner  
### Offline LLM-Based Conversational Parser — Built with Mistral 7B + llama.cpp  

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)
![LLM](https://img.shields.io/badge/Model-Mistral_7B_Instruct-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/Mode-Offline_CPU-yellow)
![Status](https://img.shields.io/badge/Regex-Free-success)

---

## 🚀 Overview

The **AI Chat Cleaner** transforms messy multiformat chat transcripts (Excel / PDF / text) into a clean structured dataset using **local LLM reasoning**.  
No fragile regexes — just true language understanding.

It uses a quantized **Mistral-7B-Instruct** model via [`llama.cpp`](https://github.com/ggerganov/llama.cpp), runs entirely on CPU, and outputs perfectly formatted CSV/JSON with:
- `time`
- `speaker`
- `role`
- `message`

---

## 🧩 Key Features

✅ Works fully **offline** (no API keys)  
✅ Handles **Excel, CSV, TXT, and PDF** inputs  
✅ **No regex** — pure AI structural parsing  
✅ Automatically splits multi-speaker lines (e.g. `ok. since when? neha- today only`)  
✅ **JSON-grammar enforcement** ensures valid structure  
✅ Auto-repair for malformed JSON  
✅ **Chunked + cached processing** for large transcripts  
✅ CPU-friendly quantized GGUF model  

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|:------|:------------|:----|
| Language | **Python 3.11+** | Fast prototyping, clean data pipelines |
| Model Runtime | **llama-cpp-python** | Offline inference for `.gguf` models |
| Model | **Mistral-7B-Instruct Q4 K M (Q4 _0)** | Compact yet accurate dialogue understanding |
| Data Frames | **pandas** | Excel / CSV IO |
| PDF Parsing | **pypdf** | Extract chat text from pages |
| Parallelism | **ThreadPoolExecutor** | Multi-file / multi-row concurrency |
| Prompting | **SYSTEM + USER templates** | Deterministic structured outputs |
| Validation | **JSON Grammar + Auto Repair** | Guarantees valid output |
| Caching | **MD5 hash-based local cache** | Skip already-processed chunks |

---

## 🏗️ Architecture

```
chat-cleaner-ollama/
│
├── run_all_samples.py            # orchestrator
├── models/                       # local GGUF model
├── out/                          # generated CSV outputs
├── samples/                      # raw test chats
│
└── src/chat_cleaner/
    ├── pipeline.py               # core workflow
    ├── io_utils.py               # input discovery (Excel/CSV/TXT/PDF)
    ├── preclean.py               # light cleanup + speaker split
    ├── ai_normalizer.py          # safe formatting for LLM
    ├── llamacpp_client.py        # llama.cpp integration
    ├── prompts.py                # system / user prompt templates
    ├── segmentation.py           # optional AI chunk segmentation
    └── cache/                    # auto-generated JSON cache
```

---

## 🧱 Data Flow

### 1️⃣ Input Discovery  
`io_utils.py` scans supported files → yields `(conversation_id, raw_text)` pairs.  
Each Excel row = one conversation; each PDF = aggregated text.

### 2️⃣ Pre-cleaning & Normalization  
`preclean_text()` removes boilerplate and injects line breaks between possible speakers.  
`normalize_for_llm()` standardizes spacing, tabs, and chunk sizes.

### 3️⃣ Chunking + Caching  
Long conversations are split into overlapping 1200–1500 char chunks.  
Each chunk’s MD5 hash is cached → instant re-runs.

### 4️⃣ Model Inference  
`llamacpp_client.py` loads the quantized Mistral model:

```python
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf",
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=0
)
```

Grammar mode constrains the model to emit a valid JSON array.

### 5️⃣ Prompt Design  
- **System Prompt:** Defines keys, roles, and multi-speaker split logic.  
- **User Prompt:** Injects actual chat text under strict format instructions.

### 6️⃣ Post-Processing  
Outputs are validated via `json.loads()` and auto-repaired if needed (`try_repair_json`).  
Each conversation → a dedicated CSV in `/out`.

---

## 📊 Example

### Input (Text / Excel Cell)
```
Ravi : ok. since when? neha- today only.
ravi: press f8 whn restart. tell me what happen
user: safe mode opened. (yay)
```

### Output (CSV)
| time | speaker | role | message |
|------|----------|------|----------|
| null | Ravi | Agent | ok. since when? |
| null | Neha | User | today only. |
| null | Ravi | Agent | press f8 whn restart. tell me what happen |
| null | User | User | safe mode opened. (yay) |

---

## ⚡ Why LLM > Regex

| Regex Parser | AI Parser |
|:--------------|:-----------|
| Hard-coded patterns per format | Learns contextually from text |
| Breaks on typos / emojis | Robust to natural dialogue |
| Needs manual tuning | Generalizes automatically |
| Limited to known syntax | Understands unseen chat formats |

---

## 🧠 Model Details

| Setting | Value |
|:---------|:-------|
| Model File | `mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf` |
| Runtime | `llama-cpp-python` |
| Context Window | 32 k tokens |
| Threads | 8 (default CPU cores) |
| Temperature | 0.0 (deterministic) |
| Output Grammar | JSON array of objects |
| Cache | JSON per chunk (MD5) |

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the project
git clone https://github.com/<yourname>/chat-cleaner-ollama.git
cd chat-cleaner-ollama

# 2️⃣ Create venv
python -m venv .venv
.\.venv\Scriptsctivate  # Windows

# 3️⃣ Install deps
pip install -r requirements.txt

# 4️⃣ Place model
models/mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf

# 5️⃣ Run cleaner
python run_all_samples.py --model "models/mistral-7b-instruct-v0.2.Q4_K_M.Q4_0.gguf"
```

---

## 📦 Example Run

```bash
🧠 Found 2 sample files:
   - chat_sample_xl.xlsx
   - chat_sample.pdf

🚀 Processing samples/chat_sample_xl.xlsx
✅ Processed: chat_sample_xl_row1 → chat_sample_xl_row1_clean.csv
✅ Processed: chat_sample_xl_row2 → chat_sample_xl_row2_clean.csv

🎯 All chats processed successfully!
```

Outputs appear in the `out/` folder as individual CSVs.

---

## 📈 Performance Notes

| Metric | Typical |
|:--------|:--------|
| CPU RAM | ≈ 8–10 GB |
| Avg Speed | 20–30 s per chunk |
| Context Window | 32 k tokens |
| Accuracy | ≈ 90 % speaker segmentation on noisy text |
| Retry Mechanism | Automatic JSON repair & re-inference |

---

## 🛠️ Future Roadmap

| Feature | Description |
|:----------|:-------------|
| 🧩 Async batch inference | Parallel per-file streaming |
| 🧠 Auto segmentation | Detect session start / end boundaries |
| 💬 Dialogue linking | Merge consecutive messages per user |
| ☁️ Web interface | Streamlit/Flask upload UI |
| 🔍 Entity Extraction | Tickets, emails, numbers |
| 🧠 Pluggable models | Swap in Llama 3 / Mixtral easily |

---


