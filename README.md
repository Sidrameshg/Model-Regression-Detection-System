# Model Regression Detection System 

A production-grade LLM quality evaluation and regression detection system. It splits quality evaluation into orthogonal signals (Attribution × Specificity × Relevance × Format Validity), incorporates an automated decision layer (ACCEPT / REVIEW / REJECT), leverages cost-aware LLM-as-judge escalation, performs non-parametric significance testing (Wilcoxon Signed-Rank with Pratt ties retention + Bootstrap), generates release readiness insights via an isolated AI analyst, and integrates directly with CI pipelines.

---

## Key Features

1. **Attribution × Specificity Split**: Isolates grounding (avoiding hallucinations) from detail density. Catches "confident hallucinations" (highly specific but ungrounded answers) that typical single-metric evaluators overlook.
2. **2D Decision Grid**: Maps quality evaluations to `ACCEPT`, `REVIEW` (with next-action suggestions like prompt retries), or `REJECT` (for confident hallucinations).
3. **Statistical Rigor**: Utilizes the non-parametric Wilcoxon Signed-Rank test (`zero_method='pratt'`) and Bootstrap confidence intervals for score comparisons, applying a Bonferroni multiple-comparisons correction to prevent false alarms.
4. **Config Hash Verification**: Records configuration weights and gates with a SHA-256 hash on every run, preventing comparisons of runs evaluated under different rules.
5. **Cost-Aware Escalation**: Skips expensive LLM calls for obviously good/bad completions; triggers the LLM-as-judge only inside the uncertain band `[0.45, 0.65]`.
6. **Thread-Safe Concurrent Runs**: Runs LLM completions in parallel threads with retry-with-backoff, collecting results in-memory and persisting them in a single SQLite transaction to avoid DB locks.
7. **Downstream AI Analyst**: Grounded strictly in statistics, an isolated AI analyst summarizes regressions, identifying what broke, the worst category, and top priority fixes.
8. **Unverified Catalog review**: Generated test cases from `catalog_generator.py` are flagged as `verified: false` and are kept out of CI blockages until manually reviewed and promoted.

---

## Project Structure

```
model-regression-detector/
├── config/
│   ├── config.yaml          # System options & model selections
│   ├── weights.yaml         # Scorer weights (sums to 1.0)
│   └── thresholds.yaml      # Decision matrix boundaries & alpha levels
├── data/
│   ├── test_cases.yaml      # Hand-authored (verified) and generated Q&A cases
│   └── source_docs/         # Source context documents for Q&A catalog generation
├── db/
│   └── eval_history.db      # SQLite run persistence
├── src/
│   ├── runner.py            # Orchestrator, mock model, parallel LLM caller
│   ├── catalog_generator.py # Q&A test generator using generator model
│   ├── scorers/
│   │   ├── attribution.py   # Grounding check (10th-percentile sentence alignment)
│   │   ├── specificity.py   # Concreteness check (entropy, content ratio, hedging)
│   │   ├── relevance.py     # Embedding cosine similarity (TF-IDF fallback)
│   │   └── format_validity.py # JSON/Table checker (returns None for free-text)
│   ├── aggregator.py        # Normalizes weights and enforces grounding floors
│   ├── decision.py          # 2D decision gate mapping & std-dev confidence
│   ├── judge.py             # LLM-as-judge (independent model, self-preference mitigation)
│   ├── analyst.py           # Downstream qualitative AI release readiness insights
│   ├── storage.py           # SQLite persistence layer and config hashing
│   ├── compare.py           # Statistical Wilcoxon comparisons & Bonferroni correction
│   └── alert.py             # Webhook alerting notifier (Slack/Discord format)
├── dashboard/
│   └── app.py               # Streamlit visualization & verification queue
├── .github/workflows/eval.yml # GitHub Actions CI regression blocker
├── main.py                  # CLI entrypoint
├── requirements.txt         # Package dependencies
└── README.md
```

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Offline Verification (Deterministic Mock Mode)
Run evaluations using mock completions designed to test specific pipeline paths:
* **Baseline Check**: Runs the high-quality mock completions model.
  ```bash
  python main.py --model mock --prompt-version v1.0
  ```
* **Subtle Quality Verification**: Verify how Wilcoxon-Pratt filters minor fluctuations.
  ```bash
  python main.py --model mock --prompt-version v1.1
  ```
* **Degraded Quality Gate Block**: Inject a major model regression to test block gates.
  ```bash
  python main.py --model degraded --prompt-version v2.0
  ```
* **Transient Exception Handling**: Verify concurrency and `run_failed`/`ERROR` exclusions.
  ```bash
  python main.py --model mock_fail --prompt-version v1.0
  ```

### 2. Auto-Generate Q&A Test Cases
To generate new unverified test cases from source docs in `data/source_docs/`:
```bash
python main.py --model mock --prompt-version v1.0 --generate-catalog
```
Generated cases are saved in `data/test_cases.yaml` with `verified: false`.

### 3. Start the Dashboard
Launch the visualization app locally to explore trends, compare runs, and review unverified cases:
```bash
streamlit run dashboard/app.py
```

---

## Quality Gates & CI

The build fails (exiting with code 1) when **either**:
* The Wilcoxon signed-rank test on `verified: true` cases shows a significant regression (p-value < Bonferroni-adjusted alpha, mean delta < 0), **or**
* Any `verified: true` case transitions to a `REJECT` decision (e.g., a safety refusal bypass or a confident hallucination).

If the `config_hash` doesn't match the baseline run, the comparison is refused before either check runs—ensuring changed scoring rules do not masquerade as model regressions.
