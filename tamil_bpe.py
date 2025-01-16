import re
import collections
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import os
import multiprocessing as mp
import numpy as np
from scipy.sparse import csr_matrix
import gzip
import json
from dataclasses import dataclass
from math import log2

@dataclass
class MergeStats:
    compression_ratio: float
    vocab_size: int
    merge_count: int

class OptimizedTamilBPE:
    def __init__(self, 
                 max_vocab_size: int = 5000,
                 min_freq: int = 5,
                 target_compression: float = 4.0,
                 block_size: int = 64 * 1024):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.target_compression = target_compression
        self.block_size = block_size
        self.vocab = {}
        self.merges = {}
        self.num_processes = mp.cpu_count() - 1
        self.common_words = set()
        self.merge_history = []
        
    def _calculate_entropy(self, freq: int, total: int) -> float:
        """Calculate entropy contribution of a token."""
        p = freq / total
        return -p * log2(p)
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        # Remove non-Tamil characters
        text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _process_block(self, block: str) -> Dict[str, int]:
        """Process a single block with more granular tokenization."""
        block = self._preprocess_text(block)
        words = block.split()
        vocab = collections.defaultdict(int)
        
        # Split all words into characters initially
        for word in words:
            # Always split into characters to allow more merge opportunities
            chars = ' '.join(list(word))
            vocab[chars] += 1
                
        return vocab
    
    def _merge_vocab_counts(self, vocab_list: List[Dict]) -> Dict:
        """Merge vocabulary counts with sparse matrix optimization."""
        merged = collections.defaultdict(int)
        for vocab in vocab_list:
            for word, count in vocab.items():
                merged[word] += count
        return dict(merged)
    
    def _get_pair_stats(self, vocab: Dict) -> Tuple[Dict, List]:
        """Get pair statistics based on frequency."""
        pair_freqs = collections.defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq

        # Return pair frequencies as a dictionary and a sorted list of pairs
        pairs_list = sorted(pair_freqs.keys(), key=lambda x: pair_freqs[x], reverse=True)
        return pair_freqs, pairs_list

    
    def _prune_vocab(self, vocab: Dict, threshold: int = None) -> Dict:
        """Prune vocabulary more aggressively."""
        if threshold is None:
            threshold = self.min_freq
        
        # Sort by frequency
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        # Always keep only top max_vocab_size items
        if len(sorted_vocab) > self.max_vocab_size:
            # Get frequency threshold from max_vocab_size position
            threshold = max(threshold, sorted_vocab[self.max_vocab_size-1][1])
            # Keep only items above threshold
            return dict(sorted_vocab[:self.max_vocab_size])
        
        return dict(sorted_vocab)
    
    def calculate_compression_ratio(self, corpus_path: str) -> float:
        """Calculate the compression ratio achieved by the current model."""
        total_chars = 0
        total_tokens = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read(self.block_size)  # Read a sample block
            text = self._preprocess_text(text)
            
            # Count original characters
            total_chars = len(text)
            
            # Count tokens after encoding
            words = text.split()
            for word in words:
                if word in self.common_words:
                    total_tokens += 1
                else:
                    chars = ' '.join(list(word))
                    # Apply merges
                    for pair, _ in self.merges.items():
                        chars = chars.replace(f"{pair[0]} {pair[1]}", 
                                           f"{pair[0]}{pair[1]}")
                    total_tokens += len(chars.split())
        
        # Avoid division by zero
        if total_tokens == 0:
            return 1.0
            
        return total_chars / total_tokens
    
    def train(self, corpus_path: str):
        print(f"Training BPE model using {self.num_processes} processes...")
        
        # Read corpus in blocks
        blocks = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            while True:
                block = f.read(self.block_size)
                if not block:
                    break
                blocks.append(block)
        
        print(f"Processing {len(blocks)} blocks...")
        
        # Process blocks in parallel
        with mp.Pool(self.num_processes) as pool:
            vocab_list = list(tqdm(
                pool.imap(self._process_block, blocks),
                total=len(blocks),
                desc="Processing blocks"
            ))
        
        # Merge vocabulary counts
        vocab = self._merge_vocab_counts(vocab_list)
        
        # Aggressive initial pruning
        vocab = self._prune_vocab(vocab, threshold=self.min_freq)
        
        initial_stats = MergeStats(
            compression_ratio=self.calculate_compression_ratio(corpus_path),
            vocab_size=len(vocab),
            merge_count=0
        )
        self.merge_history.append(initial_stats)
        
        print(f"Initial vocabulary size after pruning: {len(vocab)}")
        print("Starting merge operations...")
        
        iteration = 0
        max_iterations = 10000
        
        while iteration < max_iterations:
            # Aggressive pruning before pair statistics
            vocab = self._prune_vocab(vocab)
            
            # Get pair statistics
            pair_matrix, pairs = self._get_pair_stats(vocab)
            if not pairs:
                break
            
            # Find best pair based on frequency
            best_pair = max(pair_matrix, key=pair_matrix.get)
            best_freq = pair_matrix[best_pair]

            # Skip if merge frequency is too low
            if best_freq < self.min_freq:
                break
            
            # Perform merge
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges[best_pair] = len(self.merges)
            
            # Calculate statistics
            current_stats = MergeStats(
                compression_ratio=self.calculate_compression_ratio(corpus_path),
                vocab_size=len(vocab),
                merge_count=len(self.merges)
            )
            self.merge_history.append(current_stats)
            
            if iteration % 10 == 0:  # More frequent progress updates
                print(f"Iteration {iteration}: Compression ratio = {current_stats.compression_ratio:.2f}, "
                      f"Vocab size = {current_stats.vocab_size}")
            
            if current_stats.compression_ratio >= self.target_compression:
                print("Reached target compression ratio")
                break
            
            iteration += 1
        
        # Final pruning
        self.vocab = self._prune_vocab(vocab)
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Final compression ratio: {self.merge_history[-1].compression_ratio:.2f}")
    
    def save_model(self, prefix: str):
        """Save model with compression."""
        # Save vocabulary
        with gzip.open(f"{prefix}_vocab.json.gz", 'wt', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with gzip.open(f"{prefix}_merges.json.gz", 'wt', encoding='utf-8') as f:
            json.dump({' '.join(k): v for k, v in self.merges.items()}, 
                     f, ensure_ascii=False, indent=2)
        
        # Save merge history
        history_data = [(s.compression_ratio, s.vocab_size, s.merge_count) 
                       for s in self.merge_history]
        with gzip.open(f"{prefix}_history.json.gz", 'wt', encoding='utf-8') as f:
            json.dump(history_data, f)

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair of symbols in the vocabulary."""
        new_vocab = {}
        bigram = f"{pair[0]} {pair[1]}"
        replacement = f"{pair[0]}{pair[1]}"
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
            
        return new_vocab

if __name__ == "__main__":
    corpus_path = "tamil_wiki_text.txt"
    bpe = OptimizedTamilBPE(
        max_vocab_size=5000,
        min_freq=5,
        target_compression=4.0,
        block_size=64 * 1024
    )
    
    bpe.train(corpus_path)
    bpe.save_model("tamil_bpe_optimized") 