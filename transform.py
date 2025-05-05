import torch
from safetensors.torch import save_file
from pathlib import Path
import shutil
from qwen2_model import Transformer

# --- Configuration ---
# Directory containing the checkpoint files saved by train.py
checkpoint_dir = Path("/data2/linkdom/checkpoints/rgrpo-qwen2.5-math-1.5b-instruct")
# Path to the original Hugging Face model directory (used for config.json, tokenizer.json etc.)
original_model_path = Path("/data1/linkdom/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35") # Example path, adjust as needed
# Base directory where the safetensors models should be saved
base_output_dir = Path("/data2/linkdom/converted_model_safetensors/rgrpo-qwen2.5-math-1.5b-instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Or specify your device
# --- ---

# 1. Instantiate the model structure using the original pretrained path (Do this only once)
print(f"Loading model structure from: {original_model_path}")
model = Transformer.from_pretrained(original_model_path, device=device)
model.eval() # Set to evaluation mode

# Find all .pt files in the checkpoint directory
checkpoint_files = sorted(list(checkpoint_dir.glob("ckpt_*.pt")))
print(f"Found {len(checkpoint_files)} checkpoint files in {checkpoint_dir}")

# Loop through each checkpoint file
for checkpoint_path in checkpoint_files:
    print(f"\n--- Processing: {checkpoint_path.name} ---")

    # Define a unique output directory for this checkpoint
    checkpoint_name = checkpoint_path.stem # e.g., "ckpt_000100"
    output_dir = base_output_dir / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the state dictionary from the current checkpoint
    print(f"Loading state dict from checkpoint: {checkpoint_path}")
    # Use map_location to load onto the correct device directly
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        continue # Skip to the next file if loading fails

    # 3. Load the state dict into the model instance
    print("Applying state dict to model...")
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state_dict into model for {checkpoint_path}: {e}")
        continue # Skip to the next file if loading state_dict fails

    # 4. Save the model state dict in safetensors format
    output_safetensor_file = output_dir / "model.safetensors"
    print(f"Saving model to safetensors format: {output_safetensor_file}")
    # The state_dict is already on the correct device
    try:
        save_file(model.state_dict(), output_safetensor_file)
    except Exception as e:
        print(f"Error saving safetensors for {checkpoint_path}: {e}")
        continue # Skip to the next file if saving fails

    # 5. Copy other necessary files (tokenizer, config) for a complete HF model folder
    print("Copying tokenizer and config files...")
    files_to_copy = [
        "config.json", 
        "generation_config.json", 
        "merges.txt",
        "tokenizer_config.json", 
        "tokenizer.json", 
        "vocab.json"
    ] # Add any other relevant files
    for filename in files_to_copy:
        source_file = original_model_path / filename
        target_file = output_dir / filename
        if source_file.exists():
            try:
                shutil.copy(source_file, target_file)
            except Exception as e:
                 print(f"Error copying {filename} for {checkpoint_path}: {e}")
        else:
            print(f"Warning: Could not find {filename} in {original_model_path}")

    print(f"--- Finished processing: {checkpoint_path.name} ---")
    print(f"Model saved in Hugging Face safetensors format at: {output_dir}")


print("\nAll checkpoints processed.")
