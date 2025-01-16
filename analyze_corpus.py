import os

def analyze_corpus(corpus_path):
    """
    Analyzes the corpus file and prints various statistics
    """
    print("\n=== Corpus Analysis ===")
    
    # File size on disk
    file_size = os.path.getsize(corpus_path)
    file_size_kb = file_size / 1024
    file_size_mb = file_size / (1024 * 1024)
    print(f"\nFile size on disk:")
    print(f"- {file_size:,} bytes")
    print(f"- {file_size_kb:.2f} KB")
    print(f"- {file_size_mb:.2f} MB")
    
    # Load and analyze content
    with open(corpus_path, "r", encoding="utf-8") as f:
        # Memory size
        corpus_text = f.read()
        memory_size = len(corpus_text.encode('utf-8'))
        memory_size_kb = memory_size / 1024
        memory_size_mb = memory_size / (1024 * 1024)
        
        print(f"\nSize in memory:")
        print(f"- {memory_size:,} bytes")
        print(f"- {memory_size_kb:.2f} KB")
        print(f"- {memory_size_mb:.2f} MB")
        
        # Count statistics
        lines = corpus_text.split('\n')
        words = corpus_text.split()
        characters = len(corpus_text)
        
        print(f"\nContent statistics:")
        print(f"- Number of lines: {len(lines):,}")
        print(f"- Number of words: {len(words):,}")
        print(f"- Number of characters: {characters:,}")

if __name__ == "__main__":
    corpus_path = "tamil_corpus.txt"  # Path to your corpus file
    analyze_corpus(corpus_path) 