from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model.save_pretrained("./llama-3.2-3b-instruct")
tokenizer.save_pretrained("./llama-3.2-3b-instruct")