from functools import lru_cache
from typing import Any, Callable, List
from nltk.tokenize import word_tokenize

# ---------------------
# Helper: Split a single long word into subwords
# such that each subword's _len is <= max_chunk_size.
def split_long_word(word: str, max_chunk_size: int, _len: Callable[[str], int]) -> List[str]:
    subwords = []
    i = 0
    while i < len(word):
        # Find the largest substring starting at i with _len <= max_chunk_size.
        j = i + 1
        while j <= len(word) and _len(word[i:j]) <= max_chunk_size:
            j += 1
        # word[i:j-1] is the largest acceptable chunk.
        # If no progress is made (should not happen), force one character.
        if j - 1 == i:
            subwords.append(word[i:j])
            i = j
        else:
            subwords.append(word[i:j-1])
            i = j - 1
    return subwords

# ---------------------
# Helper: Split a long sentence into smaller pieces.
# The sentence is first split on whitespace. For any word too long,
# further split it into subwords.
def split_long_sentence(sentence: str, max_chunk_size: int, _len: Callable[[str], int]) -> List[str]:
    if _len(sentence) <= max_chunk_size:
        return [sentence]
    
    #words = sentence.split()  # simple whitespace split
    words = word_tokenize(sentence, language='russian')
    result = []
    current_line = ""
    
    for word in words:
        # If the individual word is too long, split it further.
        if _len(word) > max_chunk_size:
            subwords = split_long_word(word, max_chunk_size, _len)
        else:
            subwords = [word]
        
        for token in subwords:
            candidate = current_line + (" " if current_line else "") + token
            if _len(candidate) <= max_chunk_size:
                current_line = candidate
            else:
                if current_line:
                    result.append(current_line)
                current_line = token
    if current_line:
        result.append(current_line)
    return result

# ---------------------
# Preprocess the list of sentences: if a sentence's _len is more than max_chunk_size,
# split it further.
def preprocess_sentences(sentences: List[str], max_chunk_size: int, _len: Callable[[str], int]) -> List[str]:
    processed = []
    for sentence in sentences:
        if _len(sentence) > max_chunk_size:
            processed.extend(split_long_sentence(sentence, max_chunk_size, _len))
        else:
            processed.append(sentence)
    return processed

# ---------------------
# Main function: chunk_sentences.
# First, preprocess sentences so that none exceed max_chunk_size.
# Then, accumulate sentences into chunks (with optional overlap).
def chunk_sentences(sentences: List[str], max_chunk_size: int, overlap_size: int = 0, _len: Callable[[str], int] = len) -> List[str]:
    # Preprocess sentences to ensure none is longer than max_chunk_size.
    sentences = preprocess_sentences(sentences, max_chunk_size-overlap_size, _len)
    
    chunks = []
    current_chunk = []
    current_length = 0
    idx = 0

    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_length = _len(sentence)
        
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
            idx += 1  # move to next sentence
        else:
            if not current_chunk:
                # The sentence itself is too long (should not happen after preprocessing),
                # but in that case, add it as its own chunk.
                chunks.append(sentence)
                idx += 1
            else:
                # Finalize the current chunk.
                chunks.append(" ".join(current_chunk))
                # Compute overlap from the end of the current_chunk.
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = _len(s)
                    if overlap_length + s_len <= overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                # Try to add the problematic sentence to the overlap.
                if overlap_sentences and (overlap_length + sentence_length <= max_chunk_size):
                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length
                    idx += 1
                else:
                    # Cannot merge the sentence into the overlap; output it separately.
                    current_chunk = []
                    current_length = 0
                    chunks.append(sentence)
                    idx += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ---------------------
# Example usage:
if __name__ == "__main__":
    # Example: assume additional_content is in Russian and we obtain sentences using sent_tokenize.
    # For demonstration, we'll simulate with sample strings.
    
    # Simulated sentences (each with a known length in characters).
    s1 = "A" * 10  # length 10
    s2 = "B" * 15  # length 15
    s3 = "C" * 12  # length 12
    
    # Suppose additional_content was split into these three sentences.
    sentences = [s1, s2, s3]
    
    # Set parameters:
    max_chunk_size = 25
    overlap_size = 10
    
    # For this example, we'll use a dummy _len that is simply len.
    # In your application, _len is produced by your length_factory.
    _len = len  # Replace with your tokenizer-based _len when available.
    
    chunks = chunk_sentences(sentences, max_chunk_size, overlap_size, _len)
    
    print("Resulting chunks:")
    for chunk in chunks:
        print(repr(chunk))
