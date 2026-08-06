import numpy as np
from scipy import stats
from src.storage import get_run_results, list_runs

def run_bootstrap_ci(x, y, alpha=0.05, num_replicates=2000):
    """
    Computes a bootstrap 1-alpha confidence interval for the mean score delta (x - y).
    If the confidence interval does not contain 0.0, the change is statistically significant.
    """
    deltas = np.array(x) - np.array(y)
    n = len(deltas)
    if n == 0:
        return False, 1.0, (0.0, 0.0)
        
    if np.all(deltas == 0.0):
        return False, 1.0, (0.0, 0.0)
        
    # Bootstrap resampling
    rng = np.random.default_rng(42)  # Seed for deterministic tests
    boot_means = []
    for _ in range(num_replicates):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means.append(np.mean(sample))
        
    boot_means = np.sort(boot_means)
    low_pct = (alpha / 2.0) * 100
    high_pct = (1.0 - alpha / 2.0) * 100
    
    ci_low = np.percentile(boot_means, low_pct)
    ci_high = np.percentile(boot_means, high_pct)
    
    # Significant if 0 is not in the CI
    is_significant = (ci_low > 0.0) or (ci_high < 0.0)
    
    # Estimate a pseudo p-value from bootstrap distribution
    # (fraction of bootstrap means that cross zero, doubled for two-tailed)
    mean_delta = np.mean(deltas)
    if mean_delta > 0:
        p_val = 2.0 * np.mean(boot_means <= 0.0)
    else:
        p_val = 2.0 * np.mean(boot_means >= 0.0)
        
    p_val = min(1.0, max(0.0, p_val))
    
    return is_significant, p_val, (ci_low, ci_high)

def run_significance_test(x, y, alpha=0.05):
    """
    Runs Wilcoxon signed-rank test (Pratt method).
    Falls back to bootstrap confidence interval if Wilcoxon is invalid (e.g., small sample, all ties).
    """
    deltas = np.array(x) - np.array(y)
    
    if len(deltas) < 5 or np.all(deltas == 0.0):
        # Fallback to bootstrap for tiny samples or perfect ties
        return run_bootstrap_ci(x, y, alpha)
        
    try:
        # Pratt method keeps zero-difference pairs instead of discarding them
        stat, p_val = stats.wilcoxon(x, y, zero_method='pratt', alternative='two-sided')
        is_significant = p_val < alpha
        
        # Calculate a simple confidence interval for display
        _, _, ci = run_bootstrap_ci(x, y, alpha)
        return is_significant, p_val, ci
    except Exception:
        # Final fallback on any Wilcoxon failure
        return run_bootstrap_ci(x, y, alpha)

def compare_runs(db_path, run_id_a, run_id_b, alpha_base=0.05):
    """
    Compares two evaluation runs, running significance tests on verified cases.
    Applies Bonferroni multiple-comparisons correction across overall and category-level tests.
    """
    results_a = get_run_results(db_path, run_id_a) # Base/Previous run
    results_b = get_run_results(db_path, run_id_b) # Target/Current run
    
    if not results_a or not results_b:
        raise ValueError("One or both run IDs not found in database.")
        
    # 1. Config Hash Check
    hash_a = results_a[0].get("config_hash")
    hash_b = results_b[0].get("config_hash")
    hash_mismatch = (hash_a != hash_b)
    
    # Index results by case_id for paired comparison
    map_a = {r["case_id"]: r for r in results_a}
    map_b = {r["case_id"]: r for r in results_b}
    
    # 2. Separate verified (hand-authored) and unverified (auto-generated) cases
    # We only run significance testing and CI blockages on verified: true cases.
    paired_verified = []
    paired_unverified = []
    all_categories = set()
    
    for case_id in set(map_a.keys()).intersection(set(map_b.keys())):
        item_a = map_a[case_id]
        item_b = map_b[case_id]
        
        # Exclude failed runs from comparisons
        if item_a.get("run_failed") or item_b.get("run_failed") or item_a.get("final_score") is None or item_b.get("final_score") is None:
            continue
            
        category = item_b["category"]
        all_categories.add(category)
        
        pair = {
            "case_id": case_id,
            "category": category,
            "score_a": item_a["final_score"],
            "score_b": item_b["final_score"],
            "decision_a": item_a["decision"],
            "decision_b": item_b["decision"],
            "output_a": item_a["output_text"],
            "output_b": item_b["output_text"],
            "input_text": item_b["input_text"],
            "expected_output": item_b["expected_output"],
            "verified": bool(item_b.get("verified", 1))
        }
        
        if pair["verified"]:
            paired_verified.append(pair)
        else:
            paired_unverified.append(pair)
            
    # Calculate tests count for Bonferroni correction:
    # 1 overall test + N category tests (for categories with >= 3 verified samples)
    active_categories = []
    for cat in all_categories:
        cat_samples = [p for p in paired_verified if p["category"] == cat]
        if len(cat_samples) >= 3:
            active_categories.append(cat)
            
    # Total tests K
    num_tests = 1 + len(active_categories)
    adjusted_alpha = alpha_base / num_tests
    
    # 3. Perform Overall Significance Test on Verified
    scores_a = [p["score_a"] for p in paired_verified]
    scores_b = [p["score_b"] for p in paired_verified]
    
    overall_sig, overall_p, overall_ci = run_significance_test(scores_b, scores_a, adjusted_alpha)
    mean_diff = np.mean(scores_b) - np.mean(scores_a) if scores_b else 0.0
    
    # 4. Perform Category Significance Tests on Verified
    category_tests = {}
    for cat in active_categories:
        cat_pairs = [p for p in paired_verified if p["category"] == cat]
        cat_a = [p["score_a"] for p in cat_pairs]
        cat_b = [p["score_b"] for p in cat_pairs]
        sig, p, ci = run_significance_test(cat_b, cat_a, adjusted_alpha)
        category_tests[cat] = {
            "p_value": p,
            "is_significant": sig,
            "ci": ci,
            "mean_diff": np.mean(cat_b) - np.mean(cat_a)
        }
        
    # 5. Track Case Transitions
    transitions = []
    reject_transitions = []
    for p in paired_verified + paired_unverified:
        dec_a = p["decision_a"]
        dec_b = p["decision_b"]
        if dec_a != dec_b:
            transitions.append(p)
            if dec_b == "REJECT":
                reject_transitions.append(p)
                
    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "hash_mismatch": hash_mismatch,
        "config_hash_a": hash_a,
        "config_hash_b": hash_b,
        "mean_diff": mean_diff,
        "overall_p_value": overall_p,
        "overall_is_significant": overall_sig,
        "overall_ci": overall_ci,
        "category_results": category_tests,
        "transitions": transitions,
        "reject_transitions": reject_transitions,
        "verified_count": len(paired_verified),
        "unverified_count": len(paired_unverified),
        "adjusted_alpha": adjusted_alpha,
        "alpha_base": alpha_base
    }
