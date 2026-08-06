import os
import yaml

def load_weights(weights_path="config/weights.yaml"):
    """
    Loads scoring weights from weights.yaml.
    Validates that the weights sum to exactly 1.0 (with a small floating point tolerance).
    """
    if not os.path.exists(weights_path):
        # Default fallback weights
        return {
            "relevance": 0.3,
            "attribution": 0.4,
            "specificity": 0.2,
            "format_validity": 0.1
        }
    
    with open(weights_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # If the root is nested under 'weights' (from template variations) or direct:
    weights = data.get("weights", data)
    
    # Ensure they exist and sum to 1.0
    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Weights in {weights_path} must sum to 1.0, but sum is {total}")
        
    return weights

def load_thresholds(thresholds_path="config/thresholds.yaml"):
    """Loads thresholds from thresholds.yaml."""
    if not os.path.exists(thresholds_path):
        return {
            "attribution_floor": 0.3,
            "attribution_threshold": 0.6,
            "specificity_threshold": 0.5,
            "uncertain_zone_low": 0.45,
            "uncertain_zone_high": 0.65,
            "significance_alpha": 0.05
        }
    with open(thresholds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def aggregate_scores(scores, weights, thresholds):
    """
    Combines individual scores into a single weighted score.
    Filters out any None values (e.g. format_validity) and renormalizes weights.
    Applies a hard floor veto for low attribution (grounding).
    """
    # Filter active scores (score is not None)
    active_scores = {k: v for k, v in scores.items() if v is not None}
    
    if not active_scores:
        return 0.0
        
    # Filter and renormalize weights
    active_weights = {k: weights[k] for k in active_scores.keys() if k in weights}
    weight_sum = sum(active_weights.values())
    
    if weight_sum == 0:
        # Fallback if no weights match
        weight_sum = 1.0
        active_weights = {k: 1.0 / len(active_scores) for k in active_scores.keys()}
        
    normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}
    
    # Compute weighted sum
    aggregated_score = sum(active_scores[k] * normalized_weights[k] for k in active_scores.keys())
    
    # Apply hard floor veto on attribution
    attr_floor = thresholds.get("attribution_floor", 0.3)
    # Check if attribution was evaluated and falls below floor
    if "attribution" in active_scores and active_scores["attribution"] < attr_floor:
        # Cap the aggregated score to a maximum of 0.2 (or the floor, whichever is lower)
        aggregated_score = min(aggregated_score, 0.2)
        
    return float(aggregated_score)
