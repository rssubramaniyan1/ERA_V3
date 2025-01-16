import os
import re
import glob
from bs4 import BeautifulSoup

def extract_text_from_html(html_path):
    """
    Extracts and cleans text content from an HTML file.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text(separator=" ").strip()
            tamil_text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text)
        return tamil_text
    except UnicodeDecodeError:
        print(f"Skipping {html_path} due to encoding error")
        return ""

def create_corpus(directory, output_file):
    """
    Combines text from all HTML files into a single corpus file.
    """
    print(f"Looking for HTML files in: {directory}")
    html_files = glob.glob(os.path.join(directory, '*.html'))
    
    print(f"Found {len(html_files)} HTML files")
    
    if not html_files:
        print("No HTML files found! Please check the directory path.")
        return

    processed_files = 0
    skipped_files = 0
    
    with open(output_file, 'w', encoding='utf-8') as corpus_file:
        for html_file in html_files:
            print(f"Processing {html_file}")
            text = extract_text_from_html(html_file)
            if text:  # Only write if we got valid text
                corpus_file.write(text + '\n')
                processed_files += 1
            else:
                skipped_files += 1
                
    print(f"Corpus saved to {output_file}")
    print(f"Successfully processed {processed_files} files")
    print(f"Skipped {skipped_files} files due to errors")

if __name__ == "__main__":
    html_directory = "/Users/ravishankarsubramaniyam/Desktop/BPE/unicode_files"
    corpus_file = "tamil_corpus.txt"
    print("Starting corpus creation...")
    create_corpus(html_directory, corpus_file)

