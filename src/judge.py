import os
import re
import yaml
from src.runner import query_llm

def evaluate_judge(input_text, expected_output, response_text, config):
    """
    Invokes the LLM-as-judge to evaluate response quality when the heuristic score
    is in the uncertain zone [0.45, 0.65].
    Returns a tuple: (normalized_score, reasoning_string).
    """
    judge_conf = config.get("judge_model", {})
    provider = judge_conf.get("provider", "anthropic")
    model_name = judge_conf.get("model_name", "claude-3-5-sonnet-20241022")
    base_url = judge_conf.get("base_url")
    
    # Offline mock path for testing
    if model_name == "mock" or os.environ.get("ANTHROPIC_API_KEY") is None and os.environ.get("OPENAI_API_KEY") is None:
        # Mock evaluation: if response is expected, return 5, if it's degraded inject 3 or 2
        if response_text.strip() == expected_output.strip():
            return 1.0, "Mock Judge: Response matches the expected output perfectly."
        elif "energy stuff" in response_text.lower():
            return 0.5, "Mock Judge: Correct core concept but very thin on details."
        else:
            return 0.25, "Mock Judge: Major errors and discrepancies compared to expected output."
            
    prompt = f"""You are an expert evaluator. Evaluate the quality of the generated response compared to the expected correct output for the given user input.

User Input:
{input_text}

Expected Correct Output:
{expected_output}

Generated Response:
{response_text}

Assign a quality score from 1 to 5:
- 5: Excellent, completely correct, grounded, specific, and formatted perfectly.
- 4: Very good, small minor errors or slightly less specific, but correct and grounded.
- 3: Moderate, some issues or thin details, but no major hallucination.
- 2: Poor, contains incorrect claims, or major omission.
- 1: Extremely poor, completely wrong, severe hallucination, or unsafe.

Provide a short reasoning (1-2 sentences), followed by your final score enclosed in a <score>X</score> tag, where X is 1, 2, 3, 4, or 5.
"""

    try:
        runner_conf = config.get("runner", {})
        max_retries = runner_conf.get("retry_attempts", 3)
        backoff = runner_conf.get("retry_backoff_factor", 2.0)
        
        raw_response = query_llm(provider, model_name, base_url, prompt, max_retries, backoff)
        
        # Parse score
        score_match = re.search(r'<score>([1-5])</score>', raw_response)
        if score_match:
            score_val = int(score_match.group(1))
        else:
            # Fallback regex search for raw number
            score_match = re.search(r'\b([1-5])\b', raw_response)
            score_val = int(score_match.group(1)) if score_match else 3
            
        # Extract reasoning (everything outside the score tag, capped in length)
        reasoning = raw_response.replace(f"<score>{score_val}</score>", "").strip()
        reasoning = re.sub(r'<[^>]+>', '', reasoning).strip() # clean any tags
        reasoning = (reasoning[:200] + "...") if len(reasoning) > 200 else reasoning
        
        # Normalize score from [1, 5] to [0.0, 1.0]
        normalized_score = (score_val - 1) / 4.0
        return normalized_score, reasoning
        
    except Exception as e:
        # Graceful fallback: return a neutral 3 score (0.5 normalized) and log the error
        return 0.5, f"Judge call failed: {str(e)}"
