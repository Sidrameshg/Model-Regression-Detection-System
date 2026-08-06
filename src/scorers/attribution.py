import re
import math
from src.scorers.relevance import compute_relevance

def clean_sentence(text):
    """Cleans punctuation and whitespace from a sentence."""
    return re.sub(r'\s+', ' ', text.strip())

def split_into_sentences(text):
    """Splits text into individual sentences using punctuation boundary rules."""
    # Split on period, question mark, or exclamation mark followed by whitespace or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [clean_sentence(s) for s in raw_sentences if s.strip()]
    return cleaned

def is_boilerplate(sentence):
    """
    Checks if a sentence is conversational filler / boilerplate
    that should be excluded from semantic grounding validation.
    """
    sent_lower = sentence.lower()
    
    # Common conversational fillers/greetings/transitions
    stock_phrases = [
        "sure, here is", 
        "hope this helps", 
        "let me know if", 
        "let me know",
        "i can help with that", 
        "i cannot write", 
        "i cannot provide", 
        "result: ",
        "here is the", 
        "here is a", 
        "sure, i can", 
        "ok, here is", 
        "of course"
    ]
    
    for phrase in stock_phrases:
        if phrase in sent_lower:
            return True
            
    # Also filter out sentences with very short lengths (under 4 tokens)
    tokens = sentence.split()
    if len(tokens) < 4:
        return True
        
    return False

def percentile(data, pct):
    """
    Computes the arbitrary percentile of a list of numeric values.
    Returns 1.0 if the list is empty.
    """
    if not data:
        return 1.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1

def compute_attribution(response, expected, source_context=None):
    """
    Sentence-level semantic alignment check to detect hallucinations:
    1. Splits the response and expected text into sentences.
    2. Filters out conversational boilerplate and extremely short sentences.
    3. For each active response sentence, finds its max similarity to any expected sentence.
    4. Computes the 10th-percentile of these max-similarities.
    """
    # If the response is exactly empty, attribution is 0.0
    if not response.strip():
        return 0.0
        
    # Split into sentences
    resp_sentences = split_into_sentences(response)
    exp_sentences = split_into_sentences(expected)
    
    # If expected output has no sentences, return 1.0 (grounded in empty)
    if not exp_sentences:
        return 1.0
        
    # Filter response sentences to exclude boilerplate
    filtered_resp = [s for s in resp_sentences if not is_boilerplate(s)]
    
    # Robust fallback: if all response sentences were filtered out, keep them all
    if not filtered_resp:
        filtered_resp = resp_sentences
        
    alignments = []
    for resp_sent in filtered_resp:
        max_sim = 0.0
        for exp_sent in exp_sentences:
            sim = compute_relevance(resp_sent, exp_sent)
            if sim > max_sim:
                max_sim = sim
        alignments.append(max_sim)
        
    # We take the 10th percentile to catch localized hallucinations (i.e. even if one
    # sentence is completely ungrounded, the 10th percentile will drop significantly,
    # but normal variance in sentence paraphrasing won't trigger a total failure).
    attr_score = percentile(alignments, 10)
    return max(0.0, min(1.0, float(attr_score)))
