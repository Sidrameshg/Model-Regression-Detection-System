import argparse
import sys
import yaml
import pandas as pd
from src.runner import execute_run
from src.storage import compute_config_hash, save_run_results, get_latest_run_id, get_run_results, list_runs
from src.aggregator import load_weights, load_thresholds

def parse_args():
    parser = argparse.ArgumentParser(description="Model Regression Detection System")
    parser.add_argument("--model", type=str, required=True, help="Model version identifier (e.g. mock, claude-v1, etc.)")
    parser.add_argument("--prompt-version", type=str, required=True, help="Prompt version tag")
    parser.add_argument("--compare-with", type=str, default=None, help="Specific run ID to compare against (defaults to latest)")
    parser.add_argument("--generate-catalog", action="store_true", help="Generate new unverified test cases from source docs and exit")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configurations
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config.yaml: {e}", file=sys.stderr)
        config = {}
        
    db_path = config.get("workspace", {}).get("db_path", "db/eval_history.db")
    
    try:
        weights = load_weights()
        thresholds = load_thresholds()
    except Exception as e:
        print(f"Configuration validation error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Check if catalog generator was requested
    if args.generate_catalog:
        from src.catalog_generator import generate_test_cases_from_docs
        generate_test_cases_from_docs(config=config)
        sys.exit(0)
        
    config_hash = compute_config_hash(weights, thresholds)
    
    # 2. Identify Baseline Run
    prev_run_id = None
    if args.compare_with:
        # Check if literal run_id exists in db
        if get_run_results(db_path, args.compare_with):
            prev_run_id = args.compare_with
        else:
            # Try resolving as prompt_version
            resolved_id = get_latest_run_id(db_path, prompt_version=args.compare_with)
            if resolved_id:
                prev_run_id = resolved_id
            else:
                # Try resolving as model_version
                resolved_id = get_latest_run_id(db_path, model_version=args.compare_with)
                if resolved_id:
                    prev_run_id = resolved_id
                else:
                    # Fallback to literal
                    prev_run_id = args.compare_with
    else:
        prev_run_id = get_latest_run_id(db_path)
    
    print(f"Starting eval run for model={args.model}, prompt_version={args.prompt_version}")
    print(f"Config Hash: {config_hash}")
    
    # 3. Execute Run
    results = execute_run(args.model, args.prompt_version, config, weights, thresholds, config_hash)
    current_run_id = results[0]['run_id']
    
    # 4. Save to Database
    save_run_results(db_path, results)
    
    # 5. Compute Current Statistics
    avg_score = sum(r["final_score"] for r in results) / len(results)
    
    # Create Summary DataFrame
    df = pd.DataFrame(results)
    summary_cols = ["case_id", "category", "relevance_score", "attribution_score", "specificity_score", "final_score", "decision", "latency_ms"]
    print("\n--- Evaluation Run Summary ---")
    print(df[summary_cols].to_string(index=False))
    print("------------------------------")
    print(f"Run ID: {current_run_id}")
    print(f"Average Score: {avg_score:.4f}")
    
    # 6. Statistical Significance Comparison
    if prev_run_id:
        print(f"\nComparing current run against baseline run: {prev_run_id}")
        try:
            from src.compare import compare_runs
            comp = compare_runs(db_path, prev_run_id, current_run_id)
            
            # Print config hash check
            if comp["hash_mismatch"]:
                print(f"[WARNING] Config mismatch! Baseline run hash ({comp['config_hash_a'][:8]}) does not match current run hash ({comp['config_hash_b'][:8]}).")
                print("Comparison statistical tests will proceed but scoring definitions have changed.")
            else:
                print("[INFO] Config hashes match. Baseline and target runs use identical scoring weights.")
                
            print(f"Verified cases compared: {comp['verified_count']}")
            print(f"Score Shift (Mean delta): {comp['mean_diff']:+.4f}")
            p_display = f"{comp['overall_p_value']:.4f}" if comp['overall_p_value'] >= 0.0001 else "<0.0001"
            print(f"Overall p-value: {p_display} (Adjusted alpha: {comp['adjusted_alpha']:.4f})")
            
            if comp["overall_is_significant"]:
                sig_text = "SIGNIFICANT REGRESSION" if comp["mean_diff"] < 0 else "SIGNIFICANT IMPROVEMENT"
                print(f"Statistical Significance: {sig_text}")
            else:
                print("Statistical Significance: NO SIGNIFICANT CHANGE")
                
            # Print transitions
            if comp["transitions"]:
                print("\nDecision Transitions:")
                for t in comp["transitions"]:
                    verified_marker = "[VERIFIED]" if t["verified"] else "[UNVERIFIED]"
                    print(f"  - {t['case_id']} ({t['category']}) {verified_marker}: {t['decision_a']} -> {t['decision_b']}")
            else:
                print("\nNo decision transitions found.")
                
            # Invoke downstream AI analyst
            from src.analyst import generate_release_summary
            from src.alert import send_alert_if_needed
            
            print("\nGenerating AI Release-Readiness Analysis Summary...")
            analyst_summary = generate_release_summary(comp, config)
            print(analyst_summary)
            
            # Dispatch Slack alert if criteria met
            send_alert_if_needed(comp, analyst_summary)
            
            # Quality Gate Check (Zero-Tolerance Gates):
            # Block the build if:
            # - Config hashes match AND
            # - (A: Statistical regression detected OR B: Any verified case transitioned to REJECT)
            is_significant_regression = comp["overall_is_significant"] and comp["mean_diff"] < 0
            
            has_verified_reject_transition = False
            reject_case_ids = []
            for t in comp["transitions"]:
                if t["verified"] and t["decision_b"] == "REJECT" and t["decision_a"] != "REJECT":
                    has_verified_reject_transition = True
                    reject_case_ids.append(t["case_id"])
                    
            build_blocked = False
            block_reasons = []
            
            if not comp["hash_mismatch"]:
                if is_significant_regression:
                    build_blocked = True
                    block_reasons.append("Overall verified model quality has dropped significantly (statistical regression).")
                if has_verified_reject_transition:
                    build_blocked = True
                    block_reasons.append(f"Zero-tolerance failure: verified test case(s) {reject_case_ids} transitioned to REJECT.")
                    
            if build_blocked:
                print("\n[ERROR] CI Build Blocked! Quality gates failed:")
                for reason in block_reasons:
                    print(f"  - {reason}")
                sys.exit(1)
            else:
                print("\n[PASS] Quality gates satisfied. Build green.")
                
        except Exception as e:
            print(f"Error performing comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[INFO] No baseline run found or specified. Storing this run as the baseline.")

if __name__ == "__main__":
    main()
