# OCR Error Correction via LoRA Fine-Tuning

Fine-tunes `Qwen2.5-0.5B-Instruct` with LoRA (PEFT) to correct common OCR errors in document text (IDs, invoices, forms) — e.g. `1D card` → `ID card`, `rnodel number` → `model number`.

Downstream of an OCR engine, this kind of correction model helps clean up misread text before it's used for tasks like identity or document verification.

## Why this project

OCR pipelines frequently confuse visually similar characters (`l`/`1`/`I`, `O`/`0`, `rn`/`m`, `S`/`5`, `B`/`8`). A small, cheap-to-run LLM fine-tuned specifically for this correction task can sit as a lightweight post-processing step, rather than relying on a large general-purpose model for every document.

## What's in here

- `OCR_Correction_LoRA_Finetune.ipynb` — end-to-end Colab notebook (runs on a free T4 GPU):
  1. Builds a synthetic OCR-noise dataset by programmatically corrupting clean document-style sentences with realistic character-level OCR errors
  2. Loads `Qwen2.5-0.5B-Instruct` in 4-bit (bitsandbytes)
  3. Attaches a LoRA adapter (PEFT) over attention + MLP projections
  4. Fine-tunes with `trl`'s `SFTTrainer`
  5. Evaluates qualitatively on held-out noisy/clean pairs
  6. Saves the adapter (and shows how to push it to the Hugging Face Hub)

## Stack

PyTorch · Hugging Face `transformers` · `peft` (LoRA) · `trl` (SFTTrainer) · `bitsandbytes` (4-bit quantization) · `datasets`

## Running it

Open the notebook in Google Colab, set the runtime to a T4 GPU, and run the cells top to bottom. No external dataset download is required — the training data is generated programmatically in the notebook.

## Possible extensions

- Replace the synthetic dataset with real OCR output (run Tesseract/EasyOCR on scanned ID/invoice images, paired with ground truth)
- Report character error rate (CER) as a quantitative metric across LoRA ranks (r=8/16/32)
- Extend to a small vision-language model to correct errors directly from the document image instead of pre-OCR'd text
