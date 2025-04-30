# LLM from Scratch with OpenHermes-2.5 (Learning Exercise)

## Introduction

This project is a hands-on guide and implementation for training a small Transformer-based language model **inspired by LLMs** using the `teknium/OpenHermes-2.5` dataset.

**IMPORTANT DISCLAIMER:**
*   **"From Scratch" Definition:** This project builds the model architecture (Transformer blocks) and trains its weights from random initialization. It does **NOT** involve the massive web-scale data collection and multi-million dollar pre-training effort required for true Large Language Models like GPT-3/4 or Llama.
*   **Dataset Limitation:** `teknium/OpenHermes-2.5` is an *instruction-tuning* dataset. Training a model from random initialization *only* on this dataset will **NOT** result in a capable, general-purpose LLM. The resulting model will likely learn the conversational *format* but lack coherence, factual knowledge, and robust language understanding.
*   **Learning Goal:** The primary objective is to understand the *mechanics* of building and training a transformer: tokenization, data loading, model architecture, training loop, and basic generation.
*   **Ollama Integration:** The steps provided for Ollama conversion are *experimental* and may require significant adaptation or fail due to architectural mismatches between this simple model and standard models supported by `llama.cpp`.

## File Structure
llm_from_scratch_openhermes/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│ ├── init.py
│ ├── tokenizer_utils.py # Tokenizer logic
│ ├── model.py # Model definition
│ ├── data_utils.py # Data loading logic
│ ├── train.py # Training script
│ ├── generate.py # Inference script
│ └── config.py # (Optional) Configuration
├── scripts/
│ └── run_training.sh # Example run script
├── data/ # Placeholder for local data
├── output/ # Generated outputs
│ ├── tokenizer/
│ └── model_checkpoints/
└── ollama/
└── Modelfile # Ollama build instructions

## Prerequisites

*   Python 3.9
*   PyTorch (with CUDA support strongly recommended for reasonable training times)
*   Git
*   Access to a machine with sufficient RAM and ideally a GPU (e.g., >8GB VRAM, 16GB+ preferred).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd llm_from_scratch_openhermes
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    *   First, check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command based on your OS and CUDA version.
    *   Then, install the rest:
        ```bash
        pip install -r requirements.txt
        ```

## Usage

Follow these steps in order:

1.  **Train the Tokenizer:**
    This script processes the OpenHermes dataset to create a vocabulary and tokenizer model.
    ```bash
    python src/train_tokenizer.py
    ```
    *   This will create the `output/tokenizer/tokenizer.json` file.
    *   Configuration (vocab size, etc.) is currently within the script (`src/tokenizer_utils.py` or `src/train_tokenizer.py`).

2.  **Train the Language Model:**
    This script defines the model, loads the data, and runs the training loop, saving checkpoints.
    ```bash
    python src/train.py
    # Or use the example script: bash scripts/run_training.sh
    ```
    *   This reads the dataset and uses the tokenizer from `output/tokenizer/`.
    *   It saves model checkpoints (`.pt` files) to `output/model_checkpoints/`.
    *   Model hyperparameters (layers, dimensions) and training parameters (learning rate, epochs, batch size) are within the script (`src/model.py`, `src/train.py` or potentially `src/config.py`).
    *   **Warning:** Training can take significant time (hours/days) depending on your hardware, dataset size, and model configuration. Start with fewer epochs/smaller model for testing.

3.  **Generate Text (Inference):**
    Use the trained model to generate text based on a prompt.
    ```bash
    python src/generate.py --prompt "What is the capital of France?" --model_path output/model_checkpoints/final_model.pt
    ```
    *   Adjust `--model_path` to point to your desired checkpoint.
    *   Other options like `--max_length`, `--temperature` might be available (check `generate.py`).
    *   Remember the model's limitations – the output might not be accurate or coherent.

## Configuration

Key parameters (model dimensions, learning rate, paths, etc.) are currently set directly within the Python scripts (`src/*.py`). For more complex projects, consider consolidating these into `src/config.py`.

*   `src/tokenizer_utils.py`: Tokenizer settings (vocab size).
*   `src/model.py`: Model architecture (dimensions, layers, heads).
*   `src/train.py`: Training settings (epochs, batch size, learning rate, output paths).
*   `src/generate.py`: Inference settings (default max length, temperature).

## Output

*   `output/tokenizer/`: Contains the trained `tokenizer.json` file.
*   `output/model_checkpoints/`: Contains saved PyTorch model state dictionaries (`.pt` files) at different stages of training.

## Ollama Integration (Experimental)

The goal is to convert the trained PyTorch model (`.pt`) into the GGUF format used by Ollama via `llama.cpp`.

**Challenges:**
*   The `llama.cpp` conversion scripts expect specific model architectures (Llama, Mistral, etc.) and tensor names. Our custom `SimpleDecoderLLM` may not be directly compatible without modification.
*   This process requires installing and using the `llama.cpp` library.

**Conceptual Steps:**

1.  **Prepare Hugging Face Format (Potentially Required):** The `llama.cpp` converter often works best with models saved in the Hugging Face format. This might involve:
    *   Creating a `config.json` file describing your model architecture (mimicking a compatible format like Llama's).
    *   Saving the tokenizer using Hugging Face's `save_pretrained` method.
    *   Renaming your `.pt` file to `pytorch_model.bin`.
    *   Placing `config.json`, `tokenizer.json`, `pytorch_model.bin` etc., in a single directory.
2.  **Use `llama.cpp` Converter:**
    *   Clone `llama.cpp`: `git clone https://github.com/ggerganov/llama.cpp.git`
    *   Install its requirements: `cd llama.cpp && pip install -r requirements.txt`
    *   Run `convert.py` pointing to your prepared model directory:
        ```bash
        python convert.py /path/to/your/hf_formatted_model --outfile ./scratch_model.gguf --outtype f16
        ```
    *   *This step is the most likely to fail due to architectural mismatch.*
3.  **Quantize (Optional but Recommended):** Reduce model size.
    ```bash
    # Build quantization tool if needed: make quantize
    ./quantize ./scratch_model.gguf ./scratch_model_q4_K_M.gguf q4_k_m
    ```
4.  **Create Ollama `Modelfile`:** See the example `ollama/Modelfile` in this repository. Place the generated `.gguf` file (e.g., `scratch_model_q4_K_M.gguf`) next to the `Modelfile`. **Crucially, update the `TEMPLATE`** within the `Modelfile` to match the exact format your model was trained on (e.g., how "User:" and "Assistant:" prompts were structured in `src/data_utils.py`).
5.  **Build Ollama Model:**
    ```bash
    cd ollama # Navigate to the directory containing the Modelfile and GGUF file
    ollama create my-scratch-model -f ./Modelfile
    ```
6.  **Run with Ollama:**
    ```bash
    ollama run my-scratch-model "Your prompt here"
    ```

## Limitations

*   **Model Quality:** The model trained solely on OpenHermes-2.5 from scratch will **not** be a generally capable LLM. Expect formatting mimicry but limited coherence and knowledge.
*   **Compute Requirements:** Training even this small model requires a GPU for reasonable speed.
*   **Ollama Compatibility:** GGUF conversion for custom architectures is non-trivial.

## Future Improvements / Learning Extensions

*   Train on a much larger, more diverse dataset (like a subset of RedPajama or The Pile) for *actual* pre-training (requires massive compute).
*   Experiment with different model sizes and hyperparameters.
*   Implement more sophisticated training techniques (learning rate scheduling, different optimizers).
*   Fine-tune a pre-existing base model (like TinyLlama, Pythia, GPT-2) on OpenHermes-2.5 for better results with less compute than full pre-training.
*   Deep dive into `llama.cpp` to understand GGUF conversion requirements.
