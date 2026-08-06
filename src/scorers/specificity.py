import re
import math
import collections
from src.scorers.relevance import tokenize

# Common English function words (articles, pronouns, prepositions, conjunctions, auxiliary verbs)
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me', 
    'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 
    'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
    'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'ought', 'might', 'must', 'may'
}

# Hedging phrases to penalize (phrase-level match)
HEDGING_PHRASES = [
    "it is possible that", "it's possible that", "it could be that",
    "might be", "could be", "may be", "possibly", "probably",
    "i think", "i believe", "in my opinion", "i am not sure",
    "i'm not sure", "appears to be", "seems to", "generally speaking",
    "perhaps", "maybe", "approximate", "alleged", "supposedly"
]

def compute_entropy(tokens):
    """Computes Shannon entropy of token frequencies."""
    if not tokens:
        return 0.0
    counts = collections.Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def compute_specificity(response):
    """
    Computes a length-normalized specificity score based on:
    1. Information entropy (word diversity)
    2. Content-to-function word ratio
    3. Density of numbers/dates
    4. Penalization of hedging phrases
    """
    if not response.strip():
        return 0.0
        
    tokens = tokenize(response)
    if not tokens:
        return 0.0
        
    total_tokens = len(tokens)
    
    # 1. Information Entropy (Normalized)
    entropy = compute_entropy(tokens)
    # Entropy typically ranges from 0 (all same word) to log2(N). We normalize against typical range.
    entropy_score = min(1.0, max(0.0, (entropy - 1.0) / 6.0))
    
    # 2. Content-to-Function Word Ratio
    content_count = sum(1 for t in tokens if t not in FUNCTION_WORDS)
    content_ratio = content_count / total_tokens
    
    # 3. Numeric / Date Density
    # Match numbers (integers/floats) and dates
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    numbers = re.findall(number_pattern, response)
    # Normalize density relative to token count to prevent length bias
    numeric_density = min(1.0, len(numbers) / (total_tokens / 10.0 + 1.0))
    
    # 4. Hedging Phrases Density
    hedging_count = 0
    resp_lower = response.lower()
    for phrase in HEDGING_PHRASES:
        # Count non-overlapping occurrences of the phrase
        hedging_count += resp_lower.count(phrase)
        
    # Scale hedging penalty relative to token count
    hedging_density = min(0.8, hedging_count / (total_tokens / 20.0 + 1.0))
    
    # Weighted combination of positive signals
    raw_specificity = (0.35 * entropy_score) + (0.45 * content_ratio) + (0.20 * numeric_density)
    
    # Apply hedging penalty multiplicatively
    final_specificity = raw_specificity * (1.0 - hedging_density)
    
    return max(0.0, min(1.0, float(final_specificity)))
