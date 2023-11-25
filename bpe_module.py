import re
from collections import Counter, defaultdict
from tqdm import tqdm

def get_stats(vocab):
    """ Get counts of pairs of consecutive symbols."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """ Merge a pair of symbols to produce a new vocabulary."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def train_bpe(text, num_merges):
    # Tokenize on character level
    vocab = Counter(text)
    vocab = {' '.join(word) + ' </w>': freq for word, freq in vocab.items()}
    
    for i in tqdm(range(num_merges), desc="Training BPE"):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    
    merges = defaultdict(int)
    for word in vocab:
        symbols = word.split()
        for i in range(len(symbols) - 1):
            merges[(symbols[i], symbols[i + 1])] += 1
    
    return merges

def encode_bpe(text, merges):
    """ Encode text using the trained BPE merges."""
    symbols = list(text) + ['</w>']
    while len(symbols) > 1:
        min_pair = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in merges:
                min_pair = pair if min_pair is None else min(min_pair, pair, key=merges.get)
        if min_pair is None:
            break
        first, second = min_pair
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == first and symbols[i + 1] == second:
                symbols[i:i + 2] = [first + second]
            else:
                i += 1
    return ' '.join(symbols)

def decode_bpe(encoded_text):
    """ Decode a BPE encoded text."""
    return encoded_text.replace(' </w>', '').replace(' ', '')