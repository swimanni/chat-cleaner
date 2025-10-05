import os
import glob
import argparse
from src.chat_cleaner.pipeline import main as run_pipeline

# Supported file types
SUPPORTED_EXT = (".xlsx", ".xls", ".csv", ".txt", ".pdf")

def run_all(samples_dir: str, output_dir: str, model_path: str):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in glob.glob(os.path.join(samples_dir, "*"))
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not files:
        print(f"No supported files found in '{samples_dir}'.")
        return

    print(f"🧠 Found {len(files)} sample files:")
    for f in files:
        print(f"   - {os.path.basename(f)}")

    for file_path in files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{file_name}_clean.csv")
        print(f"\n🚀 Processing {file_path} -> {output_path}")
        run_pipeline(file_path, output_dir, model_path)

    print("\n✅ All sample files processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chat cleaner on all samples.")
    parser.add_argument("--model", required=True, help="Path to GGUF model (e.g. models/mistral-7b-instruct-v0.2.Q4_K_M.gguf)")
    parser.add_argument("--samples", default="samples", help="Samples directory")
    parser.add_argument("--output", default="out", help="Output directory")
    args = parser.parse_args()

    run_all(args.samples, args.output, args.model)
