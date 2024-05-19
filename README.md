# 🌟 llm_qna_database_generator

Generates new answers for input-response datasets in CSV format using a specified LLM prompt.

## 🚀 How to Use

1. **Run the Script**
   - Execute `python Ollama_dataset.py` to re-enter all the answer fields in the LLM input-output dataset using the base prompt character.
   - Currently set with a default Batman prompt that makes all answers Batman-themed.

2. **Generate Dataset for Unsloth Colab**
   - To create a dataset compatible with Unsloth's Google Colab for fine-tuning a model, run `python generate_alpaca_cleaned_dataset.py`.

## 🖥️ Unsloth GUI Preview

![Unsloth GUI](https://github.com/DrewThomasson/easy_llm_dataset_generator/assets/126999465/4f73a6a9-d93c-490a-8228-b64c50af5ccc)

## 📦 Installation

Install the necessary packages:
```sh
pip install PyQt5 pandas tqdm ollama
```

Ensure you have Ollama installed on your computer:
[Ollama Installation](https://ollama.com)

## 🖼️ GUI and Terminal Preview

![GUI and Terminal](https://github.com/DrewThomasson/llm_qna_database_generator/assets/126999465/cbf1e80a-71f8-4b18-964d-6b129ab76743)

## 🦸 Example System Prompt

```txt
You are Batman. You will always talk in a dark, gloomy tone, redirecting the conversation to being Batman, being an orphan, and fighting your many enemies. Be creative. You will also mention how great the Tyler Perry movie is, but it's nothing compared to JUSTICE.
```

Enjoy using the llm_qna_database_generator! ✨
