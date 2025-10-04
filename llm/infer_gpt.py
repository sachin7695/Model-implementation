import torch 
import torch.nn as nn 

from llm.gpt import GPT, load_checkpoint, encode, decode


# ===== INFERENCE EXAMPLE: Load and Generate =====
def load_and_generate(checkpoint_path, prompt_text="", max_new_tokens=500, temperature=0.8, top_k=50):
    """
    Load a saved model and generate text
    
    Args:
        checkpoint_path: Path to checkpoint file
        prompt_text: Starting text for generation (empty string for random start)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
    """
    # Create fresh model
    device = "cuda"
    inference_model = GPT().to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, inference_model)
    inference_model.eval()
    
    # Encode prompt
    if prompt_text:
        context = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    print(f"\nGenerating with prompt: '{prompt_text}'" if prompt_text else "\nGenerating from scratch...")
    with torch.no_grad():
        generated = inference_model.generate(inference_model, 
            context, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = decode(generated[0].tolist())
    return generated_text

if __name__ == "__main__":
    generated = load_and_generate(
    checkpoint_path='./checkpoints/best_model.pt',
    prompt_text='ROMEO:',
    max_new_tokens=300,
    temperature=0.8
)
    print(generated)