import collections
import os

def clean_text_for_clm(input_path, output_path, min_freq=10):
    """
    Cleans the input text for character-level language modeling.
    """

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # load the raw text 
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # convert to lowercase
    text = raw_text.lower()
    
    # define a whitelist of essential characters
    whitelist = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-\n")
    
    # 5. filter characters based on whitelist 
    cleaned_chars = []
    dropped_chars = set()

    for char in text:
        if char in whitelist:
            cleaned_chars.append(char)
        else:
            dropped_chars.add(char)

    cleaned_text = "".join(cleaned_chars)

    # save the processed dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    # output stats for the Day 1 preparation phase
    vocab_size = len(set(cleaned_text))
    print(f"--- Data Preparation Report ---")
    print(f"Final Vocabulary Size: {vocab_size} characters")
    print(f"Characters dropped: {dropped_chars if dropped_chars else 'None'}")
    print(f"Cleaned file saved to: {output_path}")

if __name__ == "__main__":
    input_file = "data/shakespeare.txt" 
    output_file = "data/shakespeare_preprocessed.txt"
    clean_text_for_clm(input_file, output_file)