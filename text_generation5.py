from transformers import pipeline
import re

# Load a pre-trained model and tokenizer
generator = pipeline('text-generation', model='gpt2')

# Prompt the user to enter their own text
user_prompt = input("Enter a detailed prompt: ")

# Generate text with adjusted parameters
result = generator(
    user_prompt,
    max_length=150,
    temperature=0.4,  # Lower value for less randomness
    top_k=30,         # Lower value for more focused outputs
    top_p=0.85,
    repetition_penalty=1.3,  # Higher value to reduce repetition
    truncation=True
)

# Extract generated text
generated_text = result[0]['generated_text']

# Advanced post-processing to clean up generated text
def clean_text(text):
    # Remove unwanted phrases
    text = re.sub(r"\b(?:inappropriate_word1|inappropriate_word2)\b", "[Filtered]", text)
    # Additional cleaning rules can be added here
    return text

# Clean the generated text
cleaned_text = clean_text(generated_text)

# Print the cleaned generated text
print(cleaned_text)
