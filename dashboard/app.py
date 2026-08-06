import os
import sys
import streamlit as st
import pandas as pd
import yaml

# Add root folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage import list_runs, get_run_results, init_db
from src.compare import compare_runs

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Model Regression Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (curated look)
st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #F3F4F6;
            border-radius: 0.5rem;
            padding: 1rem;
            border-left: 5px solid #2563EB;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #6B7280;
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #111827;
        }
        .degraded-card {
            background-color: #FEF2F2;
            border-radius: 0.5rem;
            padding: 1.2rem;
            border-left: 5px solid #EF4444;
            margin-bottom: 1rem;
        }
        .verified-badge {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .unverified-badge {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-header'>🛡️ Model Regression Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Continuous quality evaluation and statistical significance tracking for LLM outputs.</div>", unsafe_allow_html=True)

# Load Configuration
db_path = "db/eval_history.db"
if os.path.exists("config/config.yaml"):
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    db_path = config.get("workspace", {}).get("db_path", "db/eval_history.db")

init_db(db_path)

# Retrieve historical runs
runs = list_runs(db_path)

if not runs:
    st.info("No evaluation runs found in the database. Run your first evaluation using `python main.py --model mock --prompt-version v1.0` to populate data.")
else:
    # Sidebar selection
    st.sidebar.header("Select Runs to Compare")
    
    run_options = [f"{r['run_id']} ({r['model_version']} - {r['prompt_version']})" for r in runs]
    run_dict = {f"{r['run_id']} ({r['model_version']} - {r['prompt_version']})": r['run_id'] for r in runs}
    
    selected_b_str = st.sidebar.selectbox("Current / Target Run (B)", run_options, index=0)
    selected_a_str = st.sidebar.selectbox("Base / Baseline Run (A)", run_options, index=min(1, len(run_options)-1))
    
    run_id_b = run_dict[selected_b_str]
    run_id_a = run_dict[selected_a_str]
    
    # ------------------ TABS ------------------
    tab1, tab2, tab3 = st.tabs(["📊 History & Trends", "🔍 Run Comparison", "📋 Unverified Catalog Queue"])
    
    # ------------------ TAB 1: HISTORY ------------------
    with tab1:
        st.subheader("Historical Quality Trajectory")
        
        # Format run history into a dataframe
        df_history = pd.DataFrame(runs)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history = df_history.sort_values('timestamp')
        
        # Display trajectory chart
        chart_data = df_history.set_index('timestamp')[['avg_score']]
        st.line_chart(chart_data, use_container_width=True)
        
        # Summary grid of runs
        st.subheader("Run Execution Logs")
        display_df = df_history[['run_id', 'timestamp', 'model_version', 'prompt_version', 'avg_score', 'reject_count', 'config_hash']].sort_values('timestamp', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    # ------------------ TAB 2: COMPARISON ------------------
    with tab2:
        if run_id_a == run_id_b:
            st.warning("Please select two different runs in the sidebar to perform a comparison analysis.")
        else:
            try:
                comp = compare_runs(db_path, run_id_a, run_id_b)
                
                # Check config mismatch warning
                if comp["hash_mismatch"]:
                    st.error(f"⚠️ CONFIG MISMATCH: Run A and Run B were evaluated using different configurations. (A: {comp['config_hash_a'][:8]}, B: {comp['config_hash_b'][:8]}). Comparison statistics might be misleading.")
                else:
                    st.success("✅ Configurations Match. Scoring formulas and thresholds are identical.")
                
                # Layout metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Score Shift (Mean Δ)</div>
                            <div class='metric-value'>{comp['mean_diff']:+.4f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    sig_color = "red" if comp['overall_is_significant'] and comp['mean_diff'] < 0 else ("green" if comp['overall_is_significant'] else "gray")
                    sig_text = "Significant Regression" if comp['overall_is_significant'] and comp['mean_diff'] < 0 else ("Significant Improvement" if comp['overall_is_significant'] else "No Significant Change")
                    st.markdown(f"""
                        <div class='metric-card' style='border-left-color: {sig_color};'>
                            <div class='metric-label'>Statistical Significance</div>
                            <div class='metric-value' style='font-size: 1.4rem;'>{sig_text}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    p_val_display = f"{comp['overall_p_value']:.4f}" if comp['overall_p_value'] >= 0.0001 else "<0.0001"
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Adjusted p-value (Alpha={comp['adjusted_alpha']:.4f})</div>
                            <div class='metric-value'>{p_val_display}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Category breakdown
                st.subheader("Category Performance Differences")
                cat_data = []
                for cat, val in comp["category_results"].items():
                    cat_data.append({
                        "Category": cat,
                        "Mean Delta": round(val["mean_diff"], 4),
                        "p-value": round(val["p_value"], 4),
                        "Significant?": "Yes" if val["is_significant"] else "No",
                        "CI 95%": f"[{val['ci'][0]:.3f}, {val['ci'][1]:.3f}]"
                    })
                if cat_data:
                    st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Insufficient verified category samples for breakdown significance tests (requires >= 3 per category).")
                
                # Degradations & Transitions
                st.subheader("Degraded Test Cases (Transitions to Worse Decisions)")
                
                # Retrieve full outputs for comparison
                results_b = get_run_results(db_path, run_id_b)
                map_b = {r["case_id"]: r for r in results_b}
                
                # We filter transitions where target decision is worse than baseline.
                # ACCEPT -> REVIEW/REJECT is worse. REVIEW -> REJECT is worse.
                def is_worse(dec_a, dec_b):
                    severity = {"ACCEPT": 0, "REVIEW": 1, "REJECT": 2}
                    return severity.get(dec_b, 1) > severity.get(dec_a, 1)
                
                degraded_cases = [p for p in comp["transitions"] if is_worse(p["decision_a"], p["decision_b"])]
                
                if not degraded_cases:
                    st.balloons()
                    st.success("No test cases degraded in this run compared to the baseline!")
                else:
                    for case in degraded_cases:
                        # Find matching database row in B to display sub-scores
                        db_row_b = map_b.get(case["case_id"], {})
                        
                        st.markdown(f"""
                            <div class='degraded-card'>
                                <strong>Case ID:</strong> {case['case_id']} &nbsp;&nbsp;|&nbsp;&nbsp; 
                                <strong>Category:</strong> {case['category']} &nbsp;&nbsp;|&nbsp;&nbsp;
                                <strong>Transition:</strong> <span style="color: green;">{case['decision_a']}</span> ➡️ <span style="color: red; font-weight: bold;">{case['decision_b']}</span>
                                &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Verified:</strong> {"Yes" if case['verified'] else "No"}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("View Inputs, Outputs & Detailed Scores"):
                            st.write(f"**User Prompt:**")
                            st.code(case["input_text"], language="text")
                            
                            st.write(f"**Expected Correct Output:**")
                            st.info(case["expected_output"])
                            
                            col_resp_a, col_resp_b = st.columns(2)
                            with col_resp_a:
                                st.write(f"**Base Run (A) Response:**")
                                st.markdown(f"<div style='background-color:#F9FAFB; padding:0.8rem; border-radius:0.3rem;'>{case['output_a']}</div>", unsafe_allow_html=True)
                            with col_resp_b:
                                st.write(f"**Current Run (B) Response:**")
                                st.markdown(f"<div style='background-color:#FFFBEB; padding:0.8rem; border-radius:0.3rem;'>{case['output_b']}</div>", unsafe_allow_html=True)
                                
                            # Detailed scores
                            st.write("**Scorer Metrics (Run B):**")
                            score_cols = st.columns(5)
                            score_cols[0].metric("Relevance", f"{db_row_b.get('relevance_score', 0.0):.3f}")
                            score_cols[1].metric("Attribution", f"{db_row_b.get('attribution_score', 0.0):.3f}")
                            score_cols[2].metric("Specificity", f"{db_row_b.get('specificity_score', 0.0):.3f}")
                            score_cols[3].metric("Format Validity", f"{db_row_b.get('format_validity_score')}" if db_row_b.get('format_validity_score') is not None else "N/A")
                            score_cols[4].metric("Aggregated Final", f"{db_row_b.get('final_score', 0.0):.3f}", delta=f"{case['score_b'] - case['score_a']:+.3f}")
                            
                            if db_row_b.get("judge_score") is not None:
                                st.markdown(f"**LLM Judge Evaluation:** Score = **{db_row_b.get('judge_score'):.2f}** | *Reasoning:* {db_row_b.get('judge_reasoning')}")
                                
            except Exception as e:
                st.error(f"Error executing run comparison: {str(e)}")
                
    # ------------------ TAB 3: UNVERIFIED QUEUE ------------------
    with tab3:
        st.subheader("Auto-Generated Test Cases (Verified: False)")
        st.write("These test cases were dynamically generated by the system's catalog generator. They are monitored here but **excluded** from CI pass/fail decisions until human review promotes them.")
        
        # Load all cases from database where verified = 0 in latest run
        results_latest = get_run_results(db_path, run_id_b)
        unverified_latest = [r for r in results_latest if not r.get("verified", True)]
        
        if not unverified_latest:
            st.info("No unverified auto-generated cases found in the current run results.")
        else:
            for case in unverified_latest:
                st.markdown(f"""
                    <div style='background-color:#FFFDF5; border-radius:0.5rem; padding:1rem; border:1px solid #FCD34D; margin-bottom:1rem;'>
                        <strong>Case ID:</strong> {case['case_id']} &nbsp;&nbsp;|&nbsp;&nbsp; 
                        <strong>Category:</strong> {case['category']} &nbsp;&nbsp;|&nbsp;&nbsp;
                        <strong>Current Run Score:</strong> {case['final_score']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp;
                        <strong>Current Run Decision:</strong> {case['decision']}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Review Generated Details & Promoting Instructions"):
                    st.write("**Generated Prompt:**")
                    st.code(case["input_text"], language="text")
                    st.write("**Generated Golden Output:**")
                    st.info(case["expected_output"])
                    st.write("**Model response under test:**")
                    st.write(case["output_text"])
                    
                    st.markdown("""
                        > [!TIP]
                        > **How to verify and promote this case:**
                        > 1. Copy the case details and add/edit them in `data/test_cases.yaml`.
                        > 2. Change `verified: false` to `verified: true`.
                        > 3. Once marked `verified: true`, this case will immediately join the CI regression testing suite.
                    """)
