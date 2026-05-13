from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

article = """
Artificial intelligence is rapidly transforming education.
Many schools are introducing AI-powered tutoring systems
to help students learn more efficiently. Experts believe
that personalized learning will become increasingly common
in the next decade.
"""

input_text = "summarize: " + article

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

output = model.generate(
    **inputs,
    max_length=64,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\nARTICLE:\n")
print(article)

print("\nGENERATED SUMMARY:\n")
print(summary)