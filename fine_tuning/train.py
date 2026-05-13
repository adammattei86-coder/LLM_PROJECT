from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "google/flan-t5-small"

# Caricamento dataset
dataset = load_dataset("xsum")

# Tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

MAX_INPUT = 512
MAX_OUTPUT = 64

# Preprocessing
def preprocess(example):

    inputs = [
        "summarize: " + doc
        for doc in example["document"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["summary"],
        max_length=MAX_OUTPUT,
        truncation=True,
        padding="max_length"
    )

    labels_ids = labels["input_ids"]

    labels_ids = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels_ids
    ]

    model_inputs["labels"] = labels_ids

    return model_inputs


# Dataset ridotto per velocità
train_data = dataset["train"].select(range(500))
test_data = dataset["test"].select(range(50))

# Tokenizzazione
train_tokenized = train_data.map(
    preprocess,
    batched=True
)

test_tokenized = test_data.map(
    preprocess,
    batched=True
)

# Parametri training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized
)

# Training
trainer.train()

# Salvataggio modello
trainer.save_model("./fine_tuned_model")