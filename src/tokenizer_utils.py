# src/tokenizer_utils.py
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from . import config # Use relative import

def get_training_corpus(dataset_name, split):
    """Generator function to yield text data for tokenizer training."""
    print(f"Loading dataset {dataset_name} split {split} for tokenizer training...")
    # Load potentially large datasets in streaming mode if needed
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=False) # Set streaming=True for huge datasets
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure you have internet access and the dataset identifier is correct.")
        return

    print("Processing dataset text...")
    count = 0
    # Adjust iteration based on whether dataset is streaming or not
    if hasattr(dataset, '__iter__'): # Streaming dataset
         for item in dataset:
             if 'conversations' in item:
                conv = item['conversations']
                full_text = config.SOS_TOKEN + " "
                for turn in conv:
                    # Add simple role prefix if helpful, e.g., f"{turn['from']}: "
                    full_text += turn.get('value', '') + " " # Use .get for safety
                full_text += config.EOS_TOKEN
                yield full_text.strip()
                count +=1
             if count % 10000 == 0 and count > 0:
                 print(f"  Processed {count} conversations for tokenizer...")
    else: # Iterable dataset (like default load)
        buffer = []
        buffer_size = 1000 # Process in chunks to manage memory
        for i in range(0, len(dataset)):
            item = dataset[i]
            if 'conversations' in item:
                conv = item['conversations']
                full_text = config.SOS_TOKEN + " "
                for turn in conv:
                    full_text += turn.get('value', '') + " "
                full_text += config.EOS_TOKEN
                buffer.append(full_text.strip())

                if len(buffer) >= buffer_size:
                    yield from buffer
                    buffer = []
                    count += buffer_size
                    print(f"  Processed {count} conversations for tokenizer...")
        if buffer: # Yield remaining items
            yield from buffer
            count += len(buffer)
            print(f"  Processed {count} conversations for tokenizer...")


def train_or_load_tokenizer():
    """Trains a BPE tokenizer if not found, otherwise loads it."""
    tokenizer_path = os.path.join(config.TOKENIZER_SAVE_PATH, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"Tokenizer not found. Training a new one...")
        if not os.path.exists(config.TOKENIZER_SAVE_PATH):
            os.makedirs(config.TOKENIZER_SAVE_PATH)

        tokenizer = Tokenizer(BPE(unk_token=config.UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(vocab_size=config.VOCAB_SIZE_TARGET,
                            min_frequency=config.TOKENIZER_MIN_FREQUENCY,
                            special_tokens=config.SPECIAL_TOKENS_LIST)

        tokenizer.train_from_iterator(
            get_training_corpus(config.DATASET_NAME, config.DATASET_SPLIT),
            trainer=trainer
        )

        # Ensure all special tokens are definitely added after training
        num_added = tokenizer.add_special_tokens(config.SPECIAL_TOKENS_LIST)
        if num_added > 0:
             print(f"Added {num_added} special tokens explicitly after training.")


        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}")

    # Set padding and truncation after load/train
    pad_id = tokenizer.token_to_id(config.PAD_TOKEN)
    if pad_id is None:
        print(f"Warning: PAD token '{config.PAD_TOKEN}' not found in tokenizer vocab!")
        # Attempt to add it again - this might resize embedding layers later if needed
        tokenizer.add_special_tokens([config.PAD_TOKEN])
        pad_id = tokenizer.token_to_id(config.PAD_TOKEN)
        print(f"Attempted to add PAD token. New ID: {pad_id}, New vocab size: {tokenizer.get_vocab_size()}")


    tokenizer.enable_padding(pad_id=pad_id, pad_token=config.PAD_TOKEN, length=config.MAX_SEQ_LEN)
    tokenizer.enable_truncation(max_length=config.MAX_SEQ_LEN)

    # Verify essential tokens
    print(f"Tokenizer Loaded. Vocab size: {tokenizer.get_vocab_size()}")
    print(f"PAD ID: {tokenizer.token_to_id(config.PAD_TOKEN)}")
    print(f"UNK ID: {tokenizer.token_to_id(config.UNK_TOKEN)}")
    print(f"SOS ID: {tokenizer.token_to_id(config.SOS_TOKEN)}")
    print(f"EOS ID: {tokenizer.token_to_id(config.EOS_TOKEN)}")

    return tokenizer