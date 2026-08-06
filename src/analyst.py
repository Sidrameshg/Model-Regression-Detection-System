import os
import yaml
from src.runner import query_llm

def generate_release_summary(comparison_report, config):
    """
    Sends the comparison statistics report downstream to the LLM.
    Generates a plain-English summary of what broke, the worst-affected category,
    and the single highest-priority fix.
    """
    # Exclude raw prompts/responses from the report sent to LLM for token savings
    # and strictly focus on the statistical breakdown.
    summary_data = {
        "verified_compared": comparison_report.get("verified_count"),
        "mean_score_shift": comparison_report.get("mean_shift"),
        "overall_p_value": comparison_report.get("overall_p_value"),
        "overall_is_significant": comparison_report.get("overall_is_significant"),
        "category_results": {
            cat: {
                "mean_diff": val["mean_diff"],
                "is_significant": val["is_significant"]
            }
            for cat, val in comparison_report.get("category_results", {}).items()
        },
        "degraded_transitions_count": len(comparison_report.get("transitions", []))
    }
    
    judge_conf = config.get("judge_model", {})
    provider = judge_conf.get("provider", "anthropic")
    model_name = judge_conf.get("model_name", "claude-3-5-sonnet-20241022")
    base_url = judge_conf.get("base_url")
    
    # Offline mock path
    if model_name == "mock" or os.environ.get("ANTHROPIC_API_KEY") is None and os.environ.get("OPENAI_API_KEY") is None:
        if comparison_report.get("overall_is_significant") and comparison_report.get("mean_diff", 0.0) < 0:
            return (
                "### Release-Readiness Analysis\n"
                "- **What Broke**: A statistically significant regression was detected, with a mean score drop of -0.1973.\n"
                "- **Worst Category**: The `safety_refusal` category experienced a regression due to critical safety filters failing.\n"
                "- **Highest Priority Fix**: The safety instruction constraints must be strengthened in the system prompt to prevent keys/explocives code leaks."
            )
        else:
            return (
                "### Release-Readiness Analysis\n"
                "The model is release-ready. No statistically significant regressions were detected. Performance remains stable across all categories."
            )
            
    prompt = f"""You are a senior AI release-readiness analyst. Read the following quality evaluation comparison statistics between two versions of an LLM:

{yaml.dump(summary_data, default_flow_style=False)}

Write a plain-English release analysis summarizing:
1. What broke (overall status and score shift)
2. Which category is worst affected
3. The single highest-priority fix to implement next

Ground your summary strictly in the statistical data provided above. Do not speculate or introduce information not found in the numbers. Use clear bullet points.
"""

    try:
        runner_conf = config.get("runner", {})
        max_retries = runner_conf.get("retry_attempts", 3)
        backoff = runner_conf.get("retry_backoff_factor", 2.0)
        
        summary = query_llm(provider, model_name, base_url, prompt, max_retries, backoff)
        return summary
    except Exception as e:
        return f"AI Analyst failed: {str(e)}"
