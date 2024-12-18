from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=150, temperature=0.6, top_k=30, top_p=0.9, do_sample=True):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature if do_sample else None,
        top_k=top_k if do_sample else None,
        top_p=top_p if do_sample else None,
        do_sample=do_sample,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # User input for prompt
    prompt = input("Enter your prompt: ")
    
    # Parameters input from user
    max_length = int(input("Enter max length (default 150): ") or 150)
    temperature = float(input("Enter temperature (default 0.6): ") or 0.6)
    top_k = int(input("Enter top_k (default 30): ") or 30)
    top_p = float(input("Enter top_p (default 0.9): ") or 0.9)
    do_sample = input("Do sampling (default True): ").lower() in ['true', 'yes', 'y'] or True

    # Generate and print text
    print(generate_text(prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample))
