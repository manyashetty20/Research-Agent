"""Quick test - just create a valid LoRA adapter for testing without long training."""

from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = Path("lora_writer_tinyllama")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Use CPU on Mac
    torch_dtype=torch.float32,
)

print("Creating LoRA config...")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

print("Applying LoRA...")
model = get_peft_model(base_model, lora_cfg)

print(f"Saving to {output_dir}...")
output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

print("Done! LoRA adapter created and saved.")
print(f"Files: {list(output_dir.glob('*'))}")
