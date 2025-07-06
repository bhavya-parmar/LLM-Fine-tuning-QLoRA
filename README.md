# Fine-Tuning LLMs using QLoRA (Quantized Low-Rank Adaptation)

This project demonstrates how to fine-tune a large language model (**TinyLlama 1.1B Chat**) using **QLoRA**, an efficient technique that combines 4-bit quantization and Low-Rank Adaptation (LoRA). The goal is to adapt the model for better Hinglish conversational responses while staying within limited hardware constraints (e.g., Google Colab/Kaggle T4 GPU with 8–16GB VRAM).

---

### Objectives

- Fine-tune a ~1B parameter language model (TinyLlama 1.1B Chat) using QLoRA.
- Use the IndicVault dataset to adapt the model to Hinglish conversations.
- Evaluate using both quantitative (perplexity) and qualitative (response quality) methods.

---

## What is QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** is a memory and compute-efficient fine-tuning method for large language models. It enables training large models without needing powerful clusters or TPUs.

It combines:

- **4-bit Quantization:** Reduces memory usage by storing model weights in 4-bit precision using the `bitsandbytes` library.
- **LoRA (Low-Rank Adaptation):** Inserts small trainable adapter layers into the frozen base model, updating only these adapters during fine-tuning.

### Benefits of QLoRA:
- Efficient fine-tuning on 8GB–16GB VRAM GPUs
- Significantly reduced GPU memory footprint
- Faster training with minimal performance drop
- Ideal for consumer hardware or limited resource settings

---

## How QLoRA Works

1. **Quantize** the pre-trained model into 4-bit format using `bitsandbytes`.
2. **Prepare** the model with `prepare_model_for_kbit_training()` to make it LoRA-compatible.
3. **Inject LoRA adapters** using Hugging Face's `peft` library.
4. **Train only the LoRA adapters**, keeping the rest of the model frozen.
5. **Evaluate** using standard metrics and/or merge adapters with base model for deployment.

---

## Learning Outcomes

- Learn fine-tuning with Hugging Face Transformers & PEFT
- Understand LoRA-based parameter-efficient training
- Gain experience training models under resource constraints
- Evaluate models both numerically and subjectively

---
