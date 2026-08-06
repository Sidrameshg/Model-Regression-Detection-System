import os
import re
import yaml
from src.runner import query_llm

def generate_test_cases_from_docs(docs_dir="data/source_docs", test_cases_path="data/test_cases.yaml", config=None):
    """
    Scans docs_dir for documentation, calls the generator model to generate new Q&A test cases,
    and appends them to test_cases.yaml as verified: false.
    """
    if config is None:
        config = {}
        
    os.makedirs(docs_dir, exist_ok=True)
    
    # Check for doc files
    doc_files = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(('.txt', '.md'))]
    
    # 1. Read document text
    doc_text = ""
    if doc_files:
        # Read the first available file
        with open(doc_files[0], "r", encoding="utf-8") as f:
            doc_text = f.read()
    else:
        # Put a default guide file so it always has text to read
        default_file = os.path.join(docs_dir, "model_regression_guide.txt")
        doc_text = (
            "The Model Regression Detection System (MRDS) is an evaluation framework. "
            "It uses a two-axis grounding (attribution) and specificity scorer. "
            "The system is designed to catch confident hallucinations where the model output "
            "sounds very detailed and specific but contains information not present in the source correct answers. "
            "To prevent database locks on SQLite, the system aggregates all API call outputs "
            "in-memory in parallel threads, and writes them back to SQLite in a single transaction."
        )
        with open(default_file, "w", encoding="utf-8") as f:
            f.write(doc_text)
            
    # 2. Query Generator Model
    generator_conf = config.get("generator_model", {})
    provider = generator_conf.get("provider", "anthropic")
    model_name = generator_conf.get("model_name", "claude-3-5-sonnet-20241022")
    base_url = generator_conf.get("base_url")
    
    # Offline mock path
    if model_name == "mock" or os.environ.get("ANTHROPIC_API_KEY") is None and os.environ.get("OPENAI_API_KEY") is None:
        new_cases = [
            {
                "id": "generated_mrds_sqlite_locks",
                "version": 1,
                "category": "factual",
                "input": "How does the Model Regression Detection System avoid database locks on SQLite when running parallel evaluations?",
                "expected_output": "The system aggregates all API call outputs in-memory in parallel threads, and then writes them to SQLite in a single transaction in the main thread.",
                "verified": False
            },
            {
                "id": "generated_mrds_hallucinations",
                "version": 1,
                "category": "factual",
                "input": "What kind of quality errors is the Model Regression Detection System designed to catch with its two-axis scorer?",
                "expected_output": "The system is designed to catch confident hallucinations, which occur when a model response sounds detailed and specific but is not grounded in the source.",
                "verified": False
            }
        ]
    else:
        prompt = f"""You are a senior QA engineer. Given the following reference documentation, generate 2 high-quality Q&A test cases that can be used to test an LLM.

Reference Documentation Excerpt:
{doc_text}

Provide the output in valid YAML format representing a list of dicts. Each dict must contain:
- id: a unique snake_case string starting with 'generated_'
- category: 'factual' or 'reasoning'
- input: the user prompt to evaluate
- expected_output: the golden correct output grounded strictly in the documentation excerpt

Output the YAML inside a ```yaml ``` block. Do not add any other conversational text.
"""
        try:
            runner_conf = config.get("runner", {})
            max_retries = runner_conf.get("retry_attempts", 3)
            backoff = runner_conf.get("retry_backoff_factor", 2.0)
            
            raw_response = query_llm(provider, model_name, base_url, prompt, max_retries, backoff)
            
            yaml_match = re.search(r'```yaml\s*(.*?)\s*```', raw_response, re.DOTALL)
            yaml_content = yaml_match.group(1) if yaml_match else raw_response
            
            parsed = yaml.safe_load(yaml_content)
            new_cases = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "id" in item and "input" in item:
                        # Force metadata overrides
                        item["version"] = 1
                        item["verified"] = False
                        new_cases.append(item)
            else:
                raise ValueError("Parsed output is not a list of cases.")
        except Exception as e:
            print(f"[WARNING] Failed to generate catalog online: {e}. Falling back to default generated cases.")
            # Default fallback cases
            new_cases = [
                {
                    "id": "generated_mrds_sqlite_locks",
                    "version": 1,
                    "category": "factual",
                    "input": "How does the Model Regression Detection System avoid database locks on SQLite when running parallel evaluations?",
                    "expected_output": "The system aggregates all API call outputs in-memory in parallel threads, and then writes them to SQLite in a single transaction in the main thread.",
                    "verified": False
                }
            ]
            
    # 3. Append to test_cases.yaml
    # Read existing cases to prevent duplicate IDs
    existing_ids = set()
    existing_cases = []
    if os.path.exists(test_cases_path):
        with open(test_cases_path, "r", encoding="utf-8") as f:
            existing_cases = yaml.safe_load(f) or []
        existing_ids = {c["id"] for c in existing_cases if "id" in c}
        
    added_count = 0
    for nc in new_cases:
        if nc["id"] not in existing_ids:
            existing_cases.append(nc)
            added_count += 1
            
    if added_count > 0:
        with open(test_cases_path, "w", encoding="utf-8") as f:
            yaml.dump(existing_cases, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Successfully generated and appended {added_count} new unverified test cases to {test_cases_path}.")
    else:
        print("[INFO] No new unique generated test cases were added (duplicates found).")
        
    return added_count
