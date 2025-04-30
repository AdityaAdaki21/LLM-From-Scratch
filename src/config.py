# src/config.py
import torch

# --- Data and Tokenizer ---
DATASET_NAME = "teknium/OpenHermes-2.5"
# Use 'train' split, potentially add 'test' or 'validation' later
DATASET_SPLIT = 'train[:1%]' # Use a small subset for quick testing/debugging, e.g., 'train[:1%]' or 'train[:10000]'
                               # For full training, use 'train'
TOKENIZER_SAVE_PATH = "output/tokenizer" # Relative to project root
VOCAB_SIZE_TARGET = 10000 # Target size for tokenizer training
TOKENIZER_MIN_FREQUENCY = 2

# --- Special Tokens (ensure consistency) ---
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SOS_TOKEN = "[SOS]" # Start of Sequence
EOS_TOKEN = "[EOS]" # End of Sequence
MASK_TOKEN = "[MASK]" # Optional, depending on use case

SPECIAL_TOKENS_LIST = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

# --- Model Architecture ---
D_MODEL = 256        # Embedding dimension (keep small for faster training)
N_HEADS = 4          # Number of attention heads (must divide d_model)
N_LAYERS = 4         # Number of transformer blocks (keep small)
D_FF = D_MODEL * 4   # Dimension of feed-forward layer
DROPOUT = 0.1        # Dropout rate

# --- Training ---
MAX_SEQ_LEN = 512     # Max sequence length (must match tokenizer truncation if enabled)
BATCH_SIZE = 8       # Adjust based on GPU memory (start small)
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1       # Start with 1-3 for testing, increase for better results
LOG_INTERVAL = 50    # Print loss every N batches
SAVE_INTERVAL = 200  # Save checkpoint every N batches (adjust as needed)
OUTPUT_DIR = "output/model_checkpoints" # Relative to project root

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generation (Inference) ---
DEFAULT_MAX_GEN_LEN = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50