import torch
from GPT import GPT
from train import GPTConfig
import argparse
import os


def main():

    parser = argparse.ArgumentParser(description="GPT Inference")
    parser.add_argument("-ch", "--checkpoint", type=str, default="./checkpoint_best1.pt", help="Checkpoint path") # Change to your checkpoint path
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use (cpu or cuda)")
    parser.add_argument("-t", "--text", type=str, default="Cappucino is the best way to start", help="Text to generate from") # Change to your text
    parser.add_argument("-m", "--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate") # Change to your max new tokens
    parser.add_argument("-T", "--temperature", type=float, default=1.0, help="Temperature for sampling")
            # Temperature controls the randomness of the predictions
            # Lower temperature -> makes softmax sharper. The model is more confident about its predictions.
            # Higher temperature -> makes softmax flatter. The model is less confident,
            #                                  tokens have more equal probabilities.
            # Temerature = 1.0 means the model behaves normally, softmax-is
            # Usually low end is approx 0.7, high end is approx 1.2 (Copilot pred)
    parser.add_argument("-s", "--num_sentences", type=int, default=None, help="Number of sentences to generate") # Change to your number of sentences
    args = parser.parse_args()

    config = GPTConfig()
    model = GPT(config).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokens = config.tokenizer.encode(args.text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(args.device)  # Add batch dimension and move to device
    generate = model.generate(tokens, max_new_tokens=args.max_new_tokens, temperature=args.temperature, num_sentences=args.num_sentences)

    print(f"\nUsing checkpoint: {args.checkpoint}")
    print(f"Using device: {args.device}")
    print(f"Prompt: {args.text}\n\n")
    if args.num_sentences is not None:
        print(f"Generating {args.num_sentences} sentences...")
    print(f"Generating maximum {args.max_new_tokens} new tokens...\n\n")
    print("\nGenerated text:")
    print(config.tokenizer.decode(generate[0].tolist()))
    print("\n")


if __name__ == "__main__":

    main()
