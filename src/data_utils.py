# src/data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from . import config # Relative import

class HermesDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id(config.PAD_TOKEN)
        self.sos_token = config.SOS_TOKEN
        self.eos_token = config.EOS_TOKEN

        print(f"Loading and preprocessing dataset: {dataset_name} split: {split}")
        # Consider streaming=True for very large datasets during inference/fine-tuning
        # For training where shuffling is important, non-streaming is often easier unless dataset is huge
        try:
            self.dataset = load_dataset(dataset_name, split=split, streaming=False)
            self.processed_data = self._preprocess()
        except Exception as e:
             print(f"ERROR loading dataset {dataset_name} [{split}]: {e}")
             print("Please check dataset name, split, and internet connection.")
             # Handle error appropriately, maybe raise exception or return empty dataset
             self.processed_data = [] # Or raise an error

        print(f"Dataset loaded. Number of sequences: {len(self.processed_data)}")


    def _preprocess(self):
        processed = []
        skipped_count = 0
        for item in self.dataset: # Iterate directly over the loaded dataset object
             if 'conversations' not in item:
                 skipped_count += 1
                 continue

             # Construct the full text sequence including roles and special tokens
             full_text = self.sos_token + " "
             for turn in item['conversations']:
                 # Ensure 'from' and 'value' keys exist safely
                 role = turn.get('from', 'Unknown')
                 value = turn.get('value', '')
                 # Basic formatting, can be customized (e.g., ChatML <|im_start|>)
                 full_text += f"{role.capitalize()}: {value} "

             full_text += self.eos_token

             # Encode the text
             encoded = self.tokenizer.encode(full_text) # Tokenizer handles truncation based on its settings

             # Input IDs are the token IDs
             input_ids = encoded.ids

             # Skip sequences that are too short (e.g., only contain SOS/EOS)
             if len(input_ids) < 3: # Needs at least SOS, one token, EOS
                 skipped_count += 1
                 continue

             # Store as tensor
             processed.append(torch.tensor(input_ids, dtype=torch.long))

        if skipped_count > 0:
            print(f"Skipped {skipped_count} sequences (e.g., missing 'conversations' key or too short).")
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

def collate_fn(batch, pad_token_id):
    """
    Collates sequences into batches, pads them, creates labels, and attention masks.
    Args:
        batch: A list of tensors (sequences).
        pad_token_id: The ID of the padding token.
    Returns:
        A dictionary containing batched input_ids, labels, and attention_mask.
    """
    # Pad sequences to the length of the longest sequence in the batch
    sequences = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

    # Input IDs are the sequences as they are (padded)
    input_ids = sequences

    # Labels are the input_ids shifted to the left.
    # The model predicts the *next* token, so label for token at index i is token at index i+1
    # Shift labels to the left
    labels = sequences.clone()
    labels[:, :-1] = sequences[:, 1:]
    # Set the label for the last token position to pad_token_id (or -100) as there's no next token
    # Using -100 is standard practice for CrossEntropyLoss ignore_index
    labels[:, -1] = -100 # Or pad_token_id, but -100 is better for loss function

    # Replace padding token IDs in labels with -100 so they are ignored by the loss function
    labels[labels == pad_token_id] = -100

    # Create attention mask (True where padded, False otherwise)
    # This mask tells the attention mechanism NOT to attend to padding tokens.
    attention_mask = (input_ids == pad_token_id) # True for padding tokens

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# --- Helper function to create DataLoader ---
def create_dataloader(tokenizer):
    dataset = HermesDataset(
        dataset_name=config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        tokenizer=tokenizer,
        max_len=config.MAX_SEQ_LEN
    )

    pad_token_id = tokenizer.token_to_id(config.PAD_TOKEN)
    if pad_token_id is None:
        raise ValueError(f"PAD token '{config.PAD_TOKEN}' ID not found in tokenizer!")

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Shuffle for training
        # Pass pad_token_id to collate_fn using functools.partial or lambda
        collate_fn=lambda batch: collate_fn(batch, pad_token_id)
    )
    return dataloader