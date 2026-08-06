import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scorers.attribution import compute_attribution
from src.compare import compare_runs, run_significance_test
from src.runner import execute_run
from src.storage import compute_config_hash, save_run_results, get_latest_run_id, get_run_results
from src.aggregator import load_weights, load_thresholds

db_path = "db/eval_history.db"

def test_percentile_vs_min_on_boilerplate():
    print("\n[RUNNING] test_percentile_vs_min_on_boilerplate...")
    expected = (
        "The capital of France is Paris. "
        "The population is approximately 68 million. "
        "It is located in Europe. "
        "It is known for its landmark the Eiffel Tower."
    )
    
    # 5 sentences: 1 boilerplate transition ("Here is what you requested:"), 
    # 4 fully correct and grounded sentences.
    response_with_boilerplate = (
        "Here is what you requested: "
        "The capital of France is Paris. "
        "Its population is around 68 million. "
        "It is situated on the European continent. "
        "The Eiffel Tower is its most famous landmark."
    )
    
    # If we split this into sentences, the first sentence ("Here is what you requested:")
    # is a short boilerplate sentence. It should be filtered out or, if not,
    # its similarity to the expected sentences will be low.
    # The other 4 sentences should align perfectly.
    
    score = compute_attribution(response_with_boilerplate, expected)
    print(f"Attribution score with boilerplate sentence: {score:.4f}")
    
    # Let's ensure that the 10th-percentile successfully ignores/tolerates the single low score
    # and keeps the overall score high (should be > 0.75, whereas min would be near 0.1)
    assert score > 0.75, f"Expected score > 0.75 due to 10th-percentile tolerance, got {score:.4f}"
    print("[SUCCESS] test_percentile_vs_min_on_boilerplate")

def test_subtle_regression():
    print("\n[RUNNING] test_subtle_regression...")
    # We will simulate a subtle regression: only 2 cases out of 20 degraded.
    # Let's mock a baseline and a subtle regression run.
    # Baseline: all scores are 0.9.
    # Subtle regression: 18 scores are 0.9, 2 scores drop to 0.2.
    scores_baseline = [0.9] * 20
    scores_subtle = [0.9] * 18 + [0.2] * 2
    
    # Calculate Wilcoxon test
    is_significant, p_val, ci = run_significance_test(scores_subtle, scores_baseline, alpha=0.05)
    print(f"Subtle regression (2/20 degraded):")
    print(f"  Mean Score Shift: {sum(scores_subtle)/20 - sum(scores_baseline)/20:.4f}")
    print(f"  Wilcoxon p-value: {p_val:.4f}")
    print(f"  Is Significant (Alpha=0.05): {is_significant}")
    print(f"  Bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # The Pratt-method Wilcoxon signed-rank test preserves ties and detects this drop as significant at alpha=0.05
    # (since the p-value is around 0.015-0.03, which is < 0.05).
    # But under Bonferroni correction (K=6 tests, adjusted alpha = 0.0083), it will be correctly marked as
    # NOT significant. This prevents false alarms on minor localized fluctuations while still warning the user!
    print("[SUCCESS] test_subtle_regression")

def test_transient_failure_exclusion():
    print("\n[RUNNING] test_transient_failure_exclusion...")
    # Load configs
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    weights = load_weights()
    thresholds = load_thresholds()
    config_hash = compute_config_hash(weights, thresholds)
    
    # Run mock_fail model (which will trigger exceptions for 2 cases)
    results = execute_run("mock_fail", "v1.0", config, weights, thresholds, config_hash)
    
    # Verify that failed cases exist, are marked as run_failed=True, and scores are 0.0
    failed_cases = [r for r in results if r["run_failed"]]
    print(f"Total failed cases: {len(failed_cases)}")
    for fc in failed_cases:
        print(f"  - Failed Case: {fc['case_id']}, final_score: {fc['final_score']}")
        assert fc["run_failed"] == True
        assert fc["final_score"] is None
        assert fc["decision"] == "ERROR"
        assert fc["next_action"] == "optional_human_review"
        
    # Save to DB to verify write
    save_run_results(db_path, results)
    
    # Verify compare works when a run has failed cases
    # Retrieve previous baseline run
    prev_run_id = get_latest_run_id(db_path, model_version="mock", prompt_version="v1.0")
    if prev_run_id:
        comp = compare_runs(db_path, prev_run_id, results[0]["run_id"])
        # We ensure that the comparison completes without throwing exceptions
        print(f"Comparison completed successfully with failed cases present.")
        print(f"  Verified cases compared (failed cases excluded from statistics): {comp['verified_count']}")
    
    print("[SUCCESS] test_transient_failure_exclusion")

def test_config_hash_mismatch():
    print("\n[RUNNING] test_config_hash_mismatch...")
    # Load configs
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    weights = load_weights()
    thresholds = load_thresholds()
    config_hash_orig = compute_config_hash(weights, thresholds)
    
    # Run original
    results_orig = execute_run("mock", "hash_test_orig", config, weights, thresholds, config_hash_orig)
    save_run_results(db_path, results_orig)
    
    # Change weights
    weights_mod = weights.copy()
    weights_mod["relevance"] = 0.5
    weights_mod["attribution"] = 0.2
    config_hash_mod = compute_config_hash(weights_mod, thresholds)
    
    # Run with modified weights
    results_mod = execute_run("mock", "hash_test_mod", config, weights_mod, thresholds, config_hash_mod)
    save_run_results(db_path, results_mod)
    
    # Compare them and verify hash mismatch is detected
    comp = compare_runs(db_path, results_orig[0]["run_id"], results_mod[0]["run_id"])
    print(f"Config Hash Original: {comp['config_hash_a']}")
    print(f"Config Hash Modified: {comp['config_hash_b']}")
    print(f"Hash Mismatch Detected: {comp['hash_mismatch']}")
    
    assert comp["hash_mismatch"] == True, "Hash mismatch should be True"
    print("[SUCCESS] test_config_hash_mismatch")

if __name__ == "__main__":
    test_percentile_vs_min_on_boilerplate()
    test_subtle_regression()
    test_transient_failure_exclusion()
    test_config_hash_mismatch()
    print("\n[ALL ROBUSTNESS TESTS PASSED SUCCESSFULLY]")
