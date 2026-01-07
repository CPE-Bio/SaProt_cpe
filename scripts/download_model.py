import os
from transformers import EsmTokenizer, EsmForMaskedLM


# Script for downloading in local machine model of Saprot to use

def main():
    model_name = "westlake-repl/SaProt_650M_AF2"
    # Target directory based on repository structure
    output_dir = os.path.join(os.getcwd(), "weights/PLMs/SaProt_650M_AF2")
    
    print(f"Downloading {model_name} from Hugging Face Hub...")
    
    # Download and save tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    
    print(f"Saving model to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("Done! You can now use this path in your training scripts.")

if __name__ == "__main__":
    main()
