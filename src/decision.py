import math

def compute_decision_and_confidence(scores, thresholds):
    """
    Applies thresholds to map scores to ACCEPT / REVIEW / REJECT and next actions.
    Computes confidence_pct from scorer disagreement (standard deviation).
    """
    attr_thresh = thresholds.get("attribution_threshold", 0.6)
    spec_thresh = thresholds.get("specificity_threshold", 0.5)
    
    # Extract scores, defaulting if missing
    attribution = scores.get("attribution")
    specificity = scores.get("specificity")
    
    # In Stage 1, these stubs might return 1.0. Let's handle defaults:
    attr_val = attribution if attribution is not None else 1.0
    spec_val = specificity if specificity is not None else 1.0
    
    # 2D Grid Decision Logic
    is_high_attr = attr_val >= attr_thresh
    is_high_spec = spec_val >= spec_thresh
    
    if is_high_attr and is_high_spec:
        decision = "ACCEPT"
        next_action = "serve_response"
    elif is_high_attr and not is_high_spec:
        decision = "REVIEW"
        next_action = "retry_with_specific_prompt"
    elif not is_high_attr and not is_high_spec:
        decision = "REVIEW"
        next_action = "regenerate_with_grounding_prompt"
    else: # not is_high_attr and is_high_spec
        decision = "REJECT"
        next_action = "optional_human_review"
        
    # Compute confidence_pct based on standard deviation of active scores
    active_scores = [v for v in scores.values() if v is not None]
    if len(active_scores) <= 1:
        confidence_pct = 100.0
    else:
        mean_score = sum(active_scores) / len(active_scores)
        variance = sum((x - mean_score) ** 2 for x in active_scores) / len(active_scores)
        std_dev = math.sqrt(variance)
        
        # Map std_dev (which ranges from 0 to 0.5 for bounded values) to confidence
        # High std_dev -> low confidence. e.g. std_dev of 0.5 leads to 0% confidence.
        confidence_pct = max(0.0, min(100.0, 100.0 * (1.0 - 2.0 * std_dev)))
        
    return decision, next_action, round(confidence_pct, 2)
