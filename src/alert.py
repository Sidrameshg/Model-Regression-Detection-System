import os
import requests
import json

def send_alert_if_needed(comparison_report, analyst_summary):
    """
    Sends a webhook notification (Slack or Discord) if a significant quality regression
    with at least one REJECT decision is found.
    """
    # 1. Check if conditions are met
    is_regression = comparison_report.get("overall_is_significant") and comparison_report.get("mean_diff", 0.0) < 0
    
    # Check if any verified case transitioned to REJECT
    has_verified_reject = False
    for t in comparison_report.get("transitions", []):
        if t["verified"] and t["decision_b"] == "REJECT" and t["decision_a"] != "REJECT":
            has_verified_reject = True
            break
            
    if not (is_regression and has_verified_reject):
        print("[INFO] Webhook alert condition not met (requires significant regression AND verified REJECT transition).")
        return
        
    # 2. Get webhook URL
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("[INFO] Webhook alert skipped. Neither SLACK_WEBHOOK_URL nor DISCORD_WEBHOOK_URL is set in the environment.")
        return
        
    # 3. Construct Payload
    mean_shift = comparison_report.get("mean_diff", 0.0)
    p_val = comparison_report.get("overall_p_value", 1.0)
    p_display = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    
    payload = {
        "text": f"🛡️ *Model Regression Detected! CI Build Blocked.*",
        "attachments": [
            {
                "color": "#EF4444",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🛡️ *Model Regression Alert*\n*Run B (Current):* {comparison_report.get('run_id_b')}\n*Run A (Baseline):* {comparison_report.get('run_id_a')}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Metrics:*\n• *Score Shift (Mean Δ):* {mean_shift:+.4f}\n• *Overall p-value:* {p_display} (Adjusted alpha: {comparison_report.get('adjusted_alpha', 0.05):.4f})\n• *Verified Cases Compared:* {comparison_report.get('verified_count', 0)}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*AI Analyst Summary:*\n{analyst_summary}"
                        }
                    }
                ]
            }
        ]
    }
    
    # 4. Dispatch Webhook
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code in (200, 201, 204):
            print("[INFO] Webhook alert sent successfully.")
        else:
            print(f"[WARNING] Webhook alert returned status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[WARNING] Failed to send webhook alert: {e}")
