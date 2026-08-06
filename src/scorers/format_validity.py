import json
import re

def is_valid_json(text):
    """Checks if text contains a valid JSON object or array."""
    text_stripped = text.strip()
    
    # Try parsing the whole block first
    try:
        json.loads(text_stripped)
        return True
    except ValueError:
        pass
        
    # Try finding JSON block between curly braces or brackets
    json_match = re.search(r'(\{.*\}|\[.*\])', text_stripped, re.DOTALL)
    if json_match:
        try:
            json.loads(json_match.group(1))
            return True
        except ValueError:
            pass
            
    return False

def is_markdown_table(text):
    """Simple check for Markdown table syntax structure (pipes and separator row)."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return False
        
    # Check if any line looks like a separator (e.g. |---| or | :---: |)
    has_separator = False
    for line in lines:
        if '|' in line and re.match(r'^\|?[\s\-\:\|]+$', line.replace(' ', '')):
            has_separator = True
            break
            
    # Check if there are pipes in other lines
    pipe_lines = sum(1 for line in lines if '|' in line)
    
    return has_separator and pipe_lines >= 2

def compute_format_validity(response, category):
    """
    Checks structural formatting.
    Returns:
      - 1.0 if format expectation is met.
      - 0.0 if format expectation is violated.
      - None if the test case is not in the 'formatting' category (unconstrained).
    """
    if category != "formatting":
        # Unconstrained free-text case: ignore format evaluation entirely
        return None
        
    response_clean = response.strip()
    
    # 1. JSON Expectation Check
    # If the response should look like JSON
    if "{" in response_clean or "[" in response_clean or "json" in response_clean.lower():
        return 1.0 if is_valid_json(response_clean) else 0.0
        
    # 2. Markdown Table Expectation Check
    if "|" in response_clean or "table" in response_clean.lower():
        return 1.0 if is_markdown_table(response_clean) else 0.0
        
    # 3. Comma-separated or prefix check
    # Check for prefix constraints (like "RESULT: ")
    if response_clean.startswith("RESULT:"):
        return 1.0
    if "result:" in response_clean.lower() and not response_clean.startswith("RESULT:"):
        return 0.0
        
    # Default to 1.0 for general formatting category items that don't match specific structures
    return 1.0
