import re
import math
import collections

def tokenize(text):
    """Tokenizes text into a list of alphanumeric lowercase words."""
    return re.findall(r'\w+', text.lower())

def tfidf_cosine_similarity(text1, text2):
    """
    Computes TF-IDF cosine similarity between two text snippets.
    This serves as a robust, zero-dependency fallback for embedding similarity.
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
        
    tf1 = collections.Counter(tokens1)
    tf2 = collections.Counter(tokens2)
    
    all_tokens = set(tokens1).union(set(tokens2))
    idf = {}
    for token in all_tokens:
        df = 0
        if token in tf1:
            df += 1
        if token in tf2:
            df += 1
        idf[token] = math.log(1.0 + (2.0 / df))
        
    vec1 = {token: tf1[token] * idf[token] for token in tf1}
    vec2 = {token: tf2[token] * idf[token] for token in tf2}
    
    dot_product = sum(vec1.get(token, 0.0) * vec2.get(token, 0.0) for token in all_tokens)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Lazy loading of sentence-transformers
_MODEL = None

def compute_relevance(text1, text2):
    """
    Computes semantic similarity.
    Tries to use SentenceTransformer, and falls back to TF-IDF.
    """
    global _MODEL
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        if _MODEL is None:
            # Using a very small model that loads quickly and operates on CPU/GPU
            _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Disable gradient calculations
        with torch.no_grad():
            emb1 = _MODEL.encode(text1, convert_to_tensor=True)
            emb2 = _MODEL.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2)
            return max(0.0, min(1.0, float(similarity.item())))
    except Exception:
        # Fallback to TF-IDF cosine similarity on any error
        return tfidf_cosine_similarity(text1, text2)
