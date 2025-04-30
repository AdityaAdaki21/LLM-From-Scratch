# src/generate.py
import torch
import argparse
import os
from tokenizers import Tokenizer # Need direct access to load

# Use relative imports
from . import config
from .model import SimpleDecoderLLM # Import the class definition

def generate_text(prompt, model, tokenizer, device, max_length, temperature, top_k):
    """Generates text using the trained model."""
    model.eval() # Set model to evaluation mode

    # Encode the prompt, adding SOS token and potentially role markers if needed
    # Ensure this formatting matches training data input format
    formatted_prompt = f"{config.SOS_TOKEN} User: {prompt} Assistant:"
    prompt_encoded = tokenizer.encode(formatted_prompt)
    # Input needs to be batch format [1, seq_len]
    input_ids = torch.tensor([prompt_encoded.ids], dtype=torch.long).to(device)

    generated_ids = input_ids.tolist()[0] # Store generated sequence as list

    eos_token_id = tokenizer.token_to_id(config.EOS_TOKEN)
    pad_token_id = tokenizer.token_to_id(config.PAD_TOKEN)

    with torch.no_grad():
        for _ in range(max_length):
            # Get the current sequence tensor
            current_input_ids = torch.tensor([generated_ids], dtype=torch.long).to(device)
            # Create attention mask for the current sequence (mask padding if any)
            # Since we generate token by token, there's usually no padding in the input here,
            # but it's good practice if the input logic changes.
            attention_mask = (current_input_ids == pad_token_id)

            # Get logits from the model
            # Pass the attention mask (padding mask)
            outputs = model(current_input_ids, src_padding_mask=attention_mask)
            # We only care about the logits for the *next* token prediction
            next_token_logits = outputs[:, -1, :] # Shape: [1, vocab_size]

            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                 next_token_logits = next_token_logits / temperature

            # Apply Top-K sampling
            if top_k > 0:
                # Get the top_k logits and their indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # Convert logits to probabilities using softmax
                next_token_probs = torch.softmax(top_k_logits, dim=-1)
                # Sample from the filtered distribution
                # multinomial expects probabilities, returns index within the top_k set
                sampled_relative_index = torch.multinomial(next_token_probs, num_samples=1)
                # Get the actual token ID using the index from top_k_indices
                next_token_id = torch.gather(top_k_indices, -1, sampled_relative_index).item()
            else: # Greedy decoding (if temp <= 0 or top_k <= 0)
                 next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # Append the generated token ID
            generated_ids.append(next_token_id)

            # Stop if EOS token is generated
            if next_token_id == eos_token_id:
                break

    # Decode the generated IDs back to text
    # skip_special_tokens=True might be useful to remove SOS/EOS from output string
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text using the trained model.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--tokenizer_path", type=str, default=os.path.join(config.TOKENIZER_SAVE_PATH, "tokenizer.json"), help="Path to the tokenizer file.")
    parser.add_argument("--max_length", type=int, default=config.DEFAULT_MAX_GEN_LEN, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K, help="Top-K sampling K (0 to disable).")
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")

    # --- Load Tokenizer ---
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded from {args.tokenizer_path}. Vocab size: {vocab_size}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    # Instantiate the model architecture - ensure parameters match the saved model!
    # This is crucial. If the config used for training differs from the current config,
    # loading the state dict will fail. Best practice is to save config with checkpoint.
    # We'll assume the current config matches for simplicity here.
    model = SimpleDecoderLLM(
        vocab_size=vocab_size, # Get from loaded tokenizer
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_layers=config.N_LAYERS,
        dim_feedforward=config.D_FF,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT # Dropout is typically disabled by model.eval(), but architecture needs it
    )
    print(f"Loading model state dict from {args.model_path}")
    # Load the state dictionary. Use map_location to ensure it loads to the correct device.
    # Check if the checkpoint saved the whole dict or just model_state_dict
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the saved object IS the state dict
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    model.eval() # Set to evaluation mode

    print("Model loaded successfully.")

    # --- Generate Text ---
    print(f"\nGenerating text for prompt: '{args.prompt}'")
    generated_sequence = generate_text(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        device=config.DEVICE,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print("\n--- Generated Text ---")
    print(generated_sequence)

if __name__ == "__main__":
    main()