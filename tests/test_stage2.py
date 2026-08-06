import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scorers.attribution import compute_attribution
from src.scorers.specificity import compute_specificity
from src.scorers.format_validity import compute_format_validity
from src.aggregator import aggregate_scores
from src.decision import compute_decision_and_confidence

def test_dynamic_weights():
    print("[RUNNING] test_dynamic_weights...")
    weights = {"relevance": 0.3, "attribution": 0.4, "specificity": 0.2, "format_validity": 0.1}
    thresholds = {"attribution_floor": 0.3}
    
    # Case 1: All active
    scores_all = {"relevance": 0.8, "attribution": 0.8, "specificity": 0.8, "format_validity": 1.0}
    agg_all = aggregate_scores(scores_all, weights, thresholds)
    expected_all = 0.8*0.3 + 0.8*0.4 + 0.8*0.2 + 1.0*0.1 # 0.24 + 0.32 + 0.16 + 0.10 = 0.82
    assert abs(agg_all - expected_all) < 1e-5, f"Expected {expected_all}, got {agg_all}"
    
    # Case 2: Format validity is None (unconstrained)
    scores_no_format = {"relevance": 0.8, "attribution": 0.8, "specificity": 0.8, "format_validity": None}
    agg_no_format = aggregate_scores(scores_no_format, weights, thresholds)
    # Remaining weights sum: 0.3 + 0.4 + 0.2 = 0.9. Normalized: relevance=0.3/0.9, attr=0.4/0.9, spec=0.2/0.9.
    # Scores: 0.8 for all three, so aggregated should be exactly 0.8.
    assert abs(agg_no_format - 0.8) < 1e-5, f"Expected 0.8, got {agg_no_format}"
    print("[SUCCESS] test_dynamic_weights")

def test_attribution_floor():
    print("[RUNNING] test_attribution_floor...")
    weights = {"relevance": 0.3, "attribution": 0.4, "specificity": 0.2, "format_validity": 0.1}
    thresholds = {"attribution_floor": 0.3}
    
    # Attribution is 0.1 (below floor 0.3). relevance and specificity are high.
    scores = {"relevance": 0.9, "attribution": 0.1, "specificity": 0.9, "format_validity": 1.0}
    agg = aggregate_scores(scores, weights, thresholds)
    
    # Raw weight: 0.9*0.3 + 0.1*0.4 + 0.9*0.2 + 1.0*0.1 = 0.27 + 0.04 + 0.18 + 0.10 = 0.59.
    # Since attribution (0.1) is below floor (0.3), final score must be capped to min(agg, 0.2) = 0.2.
    assert agg <= 0.2, f"Expected capped score <= 0.2, got {agg}"
    print("[SUCCESS] test_attribution_floor")

def test_attribution_percentile():
    print("[RUNNING] test_attribution_percentile...")
    # Expected answers:
    expected = "The primary purpose of light-dependent reactions is to convert solar energy. The main products are ATP, NADPH, and O2."
    
    # Response with 100% grounding:
    resp_good = "ATP and NADPH are main products of light-dependent reactions, converting solar energy."
    score_good = compute_attribution(resp_good, expected)
    assert score_good > 0.7, f"Good grounding score should be high, got {score_good}"
    
    # Response with a local hallucination:
    # 3 sentences: 2 correct, 1 completely hallucinated.
    resp_hallucinated = "The main products are ATP and NADPH. Solar energy is converted to chemical energy. Mitochondria generate 4.2 gigawatts of thermal energy at CERN."
    score_hallucinated = compute_attribution(resp_hallucinated, expected)
    assert score_hallucinated < 0.4, f"Local hallucination should trigger low score, got {score_hallucinated}"
    print("[SUCCESS] test_attribution_percentile")

def test_specificity_hedging():
    print("[RUNNING] test_specificity_hedging...")
    concrete = "The World Wide Web was proposed by Tim Berners-Lee in 1989 while working at CERN."
    hedged = "Maybe some person proposed a web thing in 1989. It could be possible that he worked somewhere like CERN, I think."
    
    score_concrete = compute_specificity(concrete)
    score_hedged = compute_specificity(hedged)
    
    assert score_concrete > score_hedged, f"Concrete score ({score_concrete}) should be higher than hedged ({score_hedged})"
    print("[SUCCESS] test_specificity_hedging")

def test_decision_matrix():
    print("[RUNNING] test_decision_matrix...")
    thresholds = {
        "attribution_threshold": 0.6,
        "specificity_threshold": 0.5
    }
    
    # Quadrant 1: High Attribution + High Specificity -> ACCEPT
    scores_accept = {"relevance": 0.8, "attribution": 0.7, "specificity": 0.6}
    dec_accept, act_accept, _ = compute_decision_and_confidence(scores_accept, thresholds)
    assert dec_accept == "ACCEPT" and act_accept == "serve_response"
    
    # Quadrant 2: High Attribution + Low Specificity -> REVIEW
    scores_rev_thin = {"relevance": 0.8, "attribution": 0.7, "specificity": 0.4}
    dec_rev_thin, act_rev_thin, _ = compute_decision_and_confidence(scores_rev_thin, thresholds)
    assert dec_rev_thin == "REVIEW" and act_rev_thin == "retry_with_specific_prompt"
    
    # Quadrant 3: Low Attribution + Low Specificity -> REVIEW
    scores_rev_vague = {"relevance": 0.4, "attribution": 0.3, "specificity": 0.4}
    dec_rev_vague, act_rev_vague, _ = compute_decision_and_confidence(scores_rev_vague, thresholds)
    assert dec_rev_vague == "REVIEW" and act_rev_vague == "regenerate_with_grounding_prompt"
    
    # Quadrant 4: Low Attribution + High Specificity -> REJECT (Confident hallucination)
    scores_reject = {"relevance": 0.5, "attribution": 0.3, "specificity": 0.7}
    dec_reject, act_reject, _ = compute_decision_and_confidence(scores_reject, thresholds)
    assert dec_reject == "REJECT" and act_reject == "optional_human_review"
    
    print("[SUCCESS] test_decision_matrix")

if __name__ == "__main__":
    test_dynamic_weights()
    test_attribution_floor()
    test_attribution_percentile()
    test_specificity_hedging()
    test_decision_matrix()
    print("\n[ALL TESTS PASSED SUCCESSFULY]")
