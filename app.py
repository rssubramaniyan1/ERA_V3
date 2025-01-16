import gradio as gr
from tamil_bpe import OptimizedTamilBPE
import json
import gzip
import random
import html

def generate_color():
    """Generate a random pastel color."""
    hue = random.random()
    saturation = 0.3 + random.random() * 0.2
    value = 0.9 + random.random() * 0.1
    
    # Convert HSV to RGB
    h = hue * 6
    f = h - int(h)
    p = value * (1 - saturation)
    q = value * (1 - saturation * f)
    t = value * (1 - saturation * (1 - f))
    
    if int(h) == 0:
        r, g, b = value, t, p
    elif int(h) == 1:
        r, g, b = q, value, p
    elif int(h) == 2:
        r, g, b = p, value, t
    elif int(h) == 3:
        r, g, b = p, q, value
    elif int(h) == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def load_bpe_model(vocab_path, merges_path):
    """Load the trained BPE model and ensure proper vocabulary mapping."""
    bpe = OptimizedTamilBPE()
    
    # Load vocabulary
    with gzip.open(vocab_path, 'rt', encoding='utf-8') as f:
        vocab_data = json.load(f)
        
    # Create proper vocabulary mapping
    # Sort by frequency to ensure consistent IDs
    sorted_vocab = sorted(vocab_data.items(), key=lambda x: int(x[1]) if isinstance(x[1], str) else x[1], reverse=True)
    bpe.vocab = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
    
    # Load merges
    with gzip.open(merges_path, 'rt', encoding='utf-8') as f:
        merges_dict = json.load(f)
        bpe.merges = {tuple(k.split()): v for k, v in merges_dict.items()}
    
    return bpe

def create_html_visualization(tokens, token_ids):
    """Create HTML visualization with color-coded tokens and their IDs."""
    token_colors = {}
    html_parts = []
    
    # Add CSS style
    html_parts.append("""
    <style>
        .token-container { margin-bottom: 20px; }
        .token {
            display: inline-block;
            margin: 0.2em;
            position: relative;
        }
        .token-text {
            padding: 0.2em 0.4em;
            border-radius: 0.3em;
            display: inline-block;
        }
        .token-id {
            position: absolute;
            top: -0.8em;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.8em;
            background-color: #333;
            color: white;
            padding: 0.1em 0.3em;
            border-radius: 0.3em;
        }
        .unknown-token {
            border: 2px dashed #ff4444;
        }
    </style>
    """)
    
    # Add token sequence
    html_parts.append('<div style="margin-bottom: 20px; font-family: monospace;">')
    html_parts.append('<strong>Token IDs:</strong><br>')
    html_parts.append(f'[{", ".join(str(id) for id in token_ids)}]')
    html_parts.append('</div>')
    
    # Add tokens with IDs
    html_parts.append('<div class="token-container">')
    for token, token_id in zip(tokens, token_ids):
        if token not in token_colors:
            token_colors[token] = generate_color()
        
        color = token_colors[token]
        is_unknown = token_id >= 5000  # Adjust this threshold based on your vocabulary size
        unknown_class = ' unknown-token' if is_unknown else ''
        
        html_parts.append(
            f'<div class="token">'
            f'<span class="token-id">{token_id}</span>'
            f'<span class="token-text{unknown_class}" style="background-color: {color};">'
            f'{html.escape(token)}</span>'
            f'</div>'
        )
    html_parts.append('</div>')
    
    # Add token-ID mapping
    html_parts.append('<div style="margin-top: 20px; font-family: monospace;">')
    html_parts.append('<strong>Token-ID Mapping:</strong><br>')
    for token, token_id in zip(tokens, token_ids):
        status = " (Unknown)" if token_id >= 5000 else ""  # Adjust threshold
        html_parts.append(f'ID {token_id}: {html.escape(token)}{status}<br>')
    html_parts.append('</div>')
    
    return "".join(html_parts)

def tokenize_text(text):
    """Tokenize input text using the trained BPE model."""
    bpe = load_bpe_model('tamil_bpe_optimized_vocab.json.gz', 
                        'tamil_bpe_optimized_merges.json.gz')
    
    text = bpe._preprocess_text(text)
    words = text.split()
    
    all_tokens = []
    token_ids = []
    
    # Process each word
    for word in words:
        chars = ' '.join(list(word))
        current = chars
        
        # Apply merges in order
        for pair, _ in sorted(bpe.merges.items(), key=lambda x: x[1]):
            current = current.replace(f"{pair[0]} {pair[1]}", f"{pair[0]}{pair[1]}")
        
        word_tokens = current.split()
        all_tokens.extend(word_tokens)
        
        # Get token IDs from vocabulary
        for token in word_tokens:
            token_id = bpe.vocab.get(token, len(bpe.vocab))
            token_ids.append(token_id)
    
    # Create visualization
    visualization = create_html_visualization(all_tokens, token_ids)
    
    # Create token analysis
    token_analysis = {
        "Token Sequence": token_ids,
        "Total Tokens": len(token_ids),
        "Unique Tokens": len(set(token_ids)),
        "Token Details": [
            {
                "Position": i+1,
                "Token": token,
                "ID": id,
                "In Vocabulary": id < len(bpe.vocab)
            }
            for i, (token, id) in enumerate(zip(all_tokens, token_ids))
        ]
    }
    
    # Create statistics
    statistics = {
        "Original Characters": len(text),
        "Number of Tokens": len(token_ids),
        "Known Tokens": sum(1 for id in token_ids if id < len(bpe.vocab)),
        "Unknown Tokens": sum(1 for id in token_ids if id >= len(bpe.vocab)),
        "Compression Ratio": f"{len(text) / len(token_ids):.2f}"
    }
    
    # Return all three values separately
    return visualization, token_analysis, statistics

# Create Gradio interface
iface = gr.Interface(
    fn=tokenize_text,
    inputs=gr.Textbox(lines=5, label="Enter Tamil Text"),
    outputs=[
        gr.HTML(label="Tokenized Text"),
        gr.JSON(label="Token Analysis"),
        gr.JSON(label="Statistics")
    ],
    title="Tamil BPE Tokenizer",
    description="This app tokenizes Tamil text using a trained BPE model with a compression ratio of 4.0",
    examples=[
        ["வணக்கம் தமிழ் மொழி மிகவும் அழகானது"],
        ["கணினி அறிவியல் துறையில் செயற்கை நுண்ணறிவு முக்கியமானது"]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()