import requests
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import os

def download_tamil_wiki():
    """Download the latest Tamil Wikipedia dump."""
    # URL for the latest Tamil Wikipedia dump
    url = "https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2"
    
    print("Downloading Tamil Wikipedia dump...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open('tawiki-latest-pages-articles.xml.bz2', 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=1024), 
                        total=total_size//1024, 
                        unit='KB'):
            f.write(data)

def clean_text(text):
    """Clean Wikipedia text content."""
    # Remove XML/HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove wiki markup
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'==+.*?==+', '', text)
    text = re.sub(r"'{2,}", '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\u0B80-\u0BFF\s\.]', ' ', text)  # Keep only Tamil unicode range
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_wiki_dump():
    """Process the Wikipedia dump and extract clean Tamil text."""
    print("Processing Wikipedia dump...")
    
    output_file = 'tamil_wiki_text.txt'
    total_articles = 0
    
    with bz2.open('tawiki-latest-pages-articles.xml.bz2', 'rt', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out:
        
        # Initialize XML parsing
        context = ET.iterparse(f, events=('end',))
        
        for event, elem in tqdm(context):
            if elem.tag.endswith('page'):
                # Get namespace
                ns = elem.tag[:-4]
                
                # Find text content
                text_elem = elem.find(f'.//{ns}text')
                if text_elem is not None and text_elem.text:
                    # Clean and write text
                    clean_content = clean_text(text_elem.text)
                    if clean_content and len(clean_content.split()) > 10:  # Min 10 words
                        out.write(clean_content + '\n')
                        total_articles += 1
                
                # Clear element to save memory
                elem.clear()
    
    print(f"Processed {total_articles} articles")
    print(f"Clean text saved to {output_file}")
    
    # Calculate file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Output file size: {size_mb:.2f} MB")

if __name__ == "__main__":
    # Download if not exists
    if not os.path.exists('tawiki-latest-pages-articles.xml.bz2'):
        download_tamil_wiki()
    
    # Process the dump
    process_wiki_dump() 