import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import PromptTuningConfig, get_peft_model, TaskType
from bs4 import BeautifulSoup

# -------- Settings --------
MODEL_NAME = "google/flan-t5-small"
DATA_FILE = "combined_reduced.json"   # path to dataset file
OUTPUT_DIR = "peft_adapters/flan_t5_prompt"
MAX_INPUT_LENGTH = 512
MAX_LABEL_LENGTH = 16
NUM_VIRTUAL_TOKENS = 50
# ---------------------------

# Load dataset
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

def html_to_text(html_content):
    """Convert raw HTML to clean text."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def preprocess(batch):
    # Convert HTML → plain text
    batch["text"] = html_to_text(batch["text"])
    
    # Use longer descriptive labels
    batch["label"] = "This is a phishing website" if batch["label"] == 1 else "This is a legitimate website"
    return batch

dataset = dataset.map(preprocess)

# Train/test split
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------- Prompt Template ----------
def make_input(text):
    return (
        "You are an expert cybersecurity analyst. "
        "Classify the following webpage as phishing or legitimate.\n\n"
        f"Content:\n{text}\n\nAnswer:"
    )

def tokenize(batch):
    prompt = make_input(batch["text"])
    model_inputs = tokenizer(
        prompt, truncation=True, padding="max_length", max_length=MAX_INPUT_LENGTH
    )
    
    labels = tokenizer(
        batch["label"], truncation=True, padding="max_length", max_length=MAX_LABEL_LENGTH
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = split_dataset.map(
    tokenize, remove_columns=split_dataset["train"].column_names
)

# Print a sample to check
print("Sample tokenized input:", tokenized["train"][0])

# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Apply PEFT prompt tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,
    prompt_tuning_init="random"
)
model = get_peft_model(model, peft_config)

# Training setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_steps=20,
    fp16=False,
    save_total_limit=2,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(2000)),  # subset for quick test
    eval_dataset=tokenized["test"].select(range(200)),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save adapter and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
