import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken

# Import only the required classes
from train_gpt import GPT, GPTConfig

# Initialize model and tokenizer
def load_model():
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50257
    )
    model = GPT(config)
    model.load_state_dict(torch.load('models/gpt2_model_weights.pt', map_location='cpu'))
    model.eval()
    return model

def generate_text(prompt, max_length=30, num_sequences=5, temperature=0.8):
    # Initialize tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # Generate multiple sequences
    generated_texts = []
    for _ in range(num_sequences):
        # Tokenize input for each sequence
        input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            x = input_ids
            while x.size(1) < max_length:
                logits = model(x)[0]
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
        
        # Decode single sequence
        tokens = x[0, :].tolist()  # Use index 0 since we have one sequence
        text = enc.decode(tokens)
        generated_texts.append(text)
    
    return "\n\n".join(generated_texts)

# Load model globally
model = load_model()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter prompt", lines=3),
        gr.Slider(minimum=10, maximum=100, value=30, label="Max Length"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Sequences"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="GPT-2 Text Generator",
    description="Enter a prompt and generate text using fine-tuned GPT-2"
)

if __name__ == "__main__":
    iface.launch() 