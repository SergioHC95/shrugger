# Configuration Guide

## Setting up config.json

Before running examples or experiments, you need to create a `config.json` file in the project root.

### Quick Setup

1. Copy the template:
   ```bash
   cp config.json.template config.json
   ```

2. Edit `config.json` with your settings:
   ```json
   {
       "hf_token": "your_actual_huggingface_token",
       "model_id": "google/gemma-3-4b-it",
       "dtype": "auto"
   }
   ```

### Configuration Options

- **`hf_token`** (required): Your Hugging Face access token
  - Get one at: https://huggingface.co/settings/tokens
  - Needed to download models from Hugging Face Hub

- **`model_id`** (optional): Model identifier from Hugging Face Hub
  - Default: `"google/gemma-3-4b-it"`
  - Examples: `"meta-llama/Llama-2-7b-hf"`, `"microsoft/DialoGPT-medium"`

- **`dtype`** (optional): Data type for model inference
  - Default: `"auto"`
  - Options: `"auto"`, `"bf16"`, `"fp16"`, `"fp32"`
  - `"auto"` uses bf16 if available, otherwise fp32

### Security Note

The `config.json` file is gitignored to prevent accidentally committing your token. Never commit your actual token to version control.
