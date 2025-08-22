# Fine-Tuning Microsoft Phi-1.5b on Salesforce DialogStudio (TweetSumm) Dataset

This repository contains a Jupyter Notebook demonstrating the fine-tuning of the Microsoft Phi-1.5b model on the Salesforce DialogStudio (TweetSumm) dataset for conversation summarization tasks. The notebook uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) and 4-bit quantization via BitsAndBytes for efficient training on limited hardware (e.g., Tesla T4 GPU).

The project focuses on summarizing customer-agent conversations from Twitter support threads. The fine-tuned model generates abstractive summaries of dialogues.

## Project Overview

This notebook fine-tunes the lightweight Microsoft Phi-1.5b model (1.3B parameters) for abstractive summarization of dialogues. Key steps include:

- Loading and preprocessing the TweetSumm subset of DialogStudio.
- Applying 4-bit quantization and LoRA for efficient fine-tuning.
- Training with Supervised Fine-Tuning (SFT) using the TRL library.
- Evaluating and performing inference on test data.

The training was performed on a Google Colab environment with a Tesla T4 GPU (16GB VRAM). Only 3 training steps were executed in the notebook for demonstration, but you can adjust `max_steps` or `num_train_epochs` for full training.

**Hardware Used (from notebook output):**

- GPU: Tesla T4
- Driver: 525.105.17
- CUDA: 12.0
- Memory: 0MiB / 15360MiB used initially

**Trainable Parameters:** ~4.7M (0.33% of total parameters) after applying LoRA.

## Dataset

The dataset is the **TweetSumm** subset from Salesforce's **DialogStudio** collection, available on Hugging Face: [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio).

- **Features:** Original dialog ID, new dialog ID, dialog index, original dialog info (including summaries), log (user-agent utterances), prompt.
- **Splits:**
  - Train: 879 examples
  - Validation: 110 examples
  - Test: 110 examples
- **Preprocessing:**
  - Clean text by removing URLs, mentions, and extra spaces.
  - Format conversations as "user: [utterance]\nagent: [response]\n".
  - Use abstractive summaries from the dataset.
  - Prompt template: Instruction + Input (conversation) + Response (summary).

**Example Processed Data:**

- **Conversation:** A dialogue between a user complaining about iPhone/Apple Watch sync issues and an agent troubleshooting.
- **Summary:** "Customer enquired about his Iphone and Apple watch which is not showing his any steps/activity and health activities. Agent is asking to move to DM and look into it."
- **Full Prompt:** Includes a system prompt for summarization.

The dataset is shuffled (seed=42) and columns are removed post-processing.

## Model

- **Base Model:** [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) (1.3B parameters, causal LM).
- **Quantization:** 4-bit (NF4 type, double quantization, compute dtype=float16) using BitsAndBytes.
- **PEFT (LoRA):**
  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0.05
  - Target Modules: ["Wqkv", "out_proj"]
  - Bias: None
  - Task Type: CAUSAL_LM
- **Tokenizer:** CodeGenTokenizerFast with pad_token set to eos_token.

The model is loaded with `trust_remote_code=True` and `use_cache=False` during training.

## Requirements

- Python 3.10+
- GPU with CUDA support (tested on Tesla T4)
- Hugging Face account (for model/dataset access and pushing to Hub)

### Data Preparation

- Load dataset: `load_dataset("Salesforce/dialogstudio", "TweetSumm")`
- Process: Shuffle, generate prompts, remove unnecessary columns.

### Model Setup

- Create model/tokenizer with 4-bit quantization.
- Apply LoRA via `get_peft_model`.

### Training

- **Arguments:**
  - Output Dir: "phi-1_5-finetuned-dialogstudio"
  - Batch Size: 4 (per device)
  - Gradient Accumulation: 1
  - Learning Rate: 2e-4 (cosine scheduler)
  - Max Steps: 3 (demo; increase for full training)
  - Epochs: 1
  - Logging: Every step
  - Save: Per epoch
  - Push to Hub: True
- Trainer: `SFTTrainer` with dataset_text_field="text".
- Run: `trainer.train()`

**Training Output (from notebook):**

- Steps: 3
- Losses: 3.0808, 2.8871, 3.3027
- Runtime: ~7.77s
- FLOPs: 22.4T

View logs with TensorBoard:

```bash
%tensorboard --logdir experiments/runs
```

### Evaluation

- Run: `trainer.evaluate()`
- **Output (from notebook):**
  - Eval Loss: 2.853
  - Runtime: ~10.67s
  - Samples/sec: 10.31
  - Steps/sec: 2.62

### Inference

- Load fine-tuned model:
  ```python
  from peft import PeftModel
  model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5"), "kingabzpro/phi-1_5-finetuned-dialogstudio")
  ```
- Example Inference:
  ```python
  def generate_inference_prompt(conversation: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
      return f"""### Instruction: {system_prompt}

  ### Input:
  {conversation.strip()}

  ### Response:
  """.strip()

  example_text = "user: Twitter, you suspended my account despite...\nagent: Please DM us your @username..."
  inference_prompt = generate_inference_prompt(example_text)

  inputs = tokenizer(inference_prompt, return_tensors="pt").to(DEVICE)
  outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95)
  summary = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[1].strip()
  print(summary)
  ```
- **Sample Output:** A generated summary of the conversation.

## Results

- The model (after minimal training) generates coherent summaries.
- Full training (increase max_steps) would improve quality.
- Evaluation Loss: 2.853 (perplexity ~17.35).
- Inference Example: Summarizes a conversation about account suspension, suggesting agent assistance via DM.

## Limitations and Notes

- Training was limited to 3 steps for demo; full fine-tuning requires more steps/epochs.
- Quantization reduces precision; use 8-bit or full precision for better results if hardware allows.
- Dataset is Twitter-specific; may not generalize to other dialogues.
- No internet access in code interpreter; all deps are pre-installed.
- Potential Errors: Ensure Hugging Face login for gated access.
