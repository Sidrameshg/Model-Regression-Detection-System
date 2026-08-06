import sqlite3
import json
import hashlib
import os

def compute_config_hash(weights_dict, thresholds_dict):
    """
    Computes a SHA-256 hash of the weights and thresholds configurations
    to ensure consistency across comparison runs.
    """
    # Normalize by sorting keys
    weights_json = json.dumps(weights_dict, sort_keys=True)
    thresholds_json = json.dumps(thresholds_dict, sort_keys=True)
    combined = f"{weights_json}||{thresholds_json}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def init_db(db_path):
    """
    Initializes the SQLite database and ensures the runs table exists.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT,
            model_version TEXT,
            prompt_version TEXT,
            case_id TEXT,
            case_version INTEGER,
            category TEXT,
            input_text TEXT,
            expected_output TEXT,
            output_text TEXT,
            latency_ms REAL,
            relevance_score REAL,
            attribution_score REAL,
            specificity_score REAL,
            format_validity_score REAL,
            aggregated_score REAL,
            judge_score REAL,
            judge_reasoning TEXT,
            final_score REAL,
            decision TEXT,
            next_action TEXT,
            confidence_pct REAL,
            config_hash TEXT,
            run_failed INTEGER,
            verified INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

def save_run_results(db_path, results):
    """
    Saves a batch of test run results in a single thread-safe database transaction.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO runs (
            run_id, timestamp, model_version, prompt_version,
            case_id, case_version, category, input_text, expected_output,
            output_text, latency_ms, relevance_score, attribution_score,
            specificity_score, format_validity_score, aggregated_score,
            judge_score, judge_reasoning, final_score, decision, next_action,
            confidence_pct, config_hash, run_failed, verified
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    data = []
    for r in results:
        data.append((
            r.get("run_id"),
            r.get("timestamp"),
            r.get("model_version"),
            r.get("prompt_version"),
            r.get("case_id"),
            r.get("case_version"),
            r.get("category"),
            r.get("input_text"),
            r.get("expected_output"),
            r.get("output_text"),
            r.get("latency_ms"),
            r.get("relevance_score"),
            r.get("attribution_score"),
            r.get("specificity_score"),
            r.get("format_validity_score"),
            r.get("aggregated_score"),
            r.get("judge_score"),
            r.get("judge_reasoning"),
            r.get("final_score"),
            r.get("decision"),
            r.get("next_action"),
            r.get("confidence_pct"),
            r.get("config_hash"),
            1 if r.get("run_failed", False) else 0,
            1 if r.get("verified", True) else 0
        ))
        
    cursor.executemany(insert_query, data)
    conn.commit()
    conn.close()

def get_run_results(db_path, run_id):
    """
    Retrieves all records for a given run_id.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_latest_run_id(db_path, model_version=None, prompt_version=None):
    """
    Retrieves the most recent run_id in the database.
    Optionally filters by model_version or prompt_version.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if model_version and prompt_version:
        cursor.execute("""
            SELECT run_id FROM runs 
            WHERE model_version = ? AND prompt_version = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (model_version, prompt_version))
    elif model_version:
        cursor.execute("SELECT run_id FROM runs WHERE model_version = ? ORDER BY timestamp DESC LIMIT 1", (model_version,))
    elif prompt_version:
        cursor.execute("SELECT run_id FROM runs WHERE prompt_version = ? ORDER BY timestamp DESC LIMIT 1", (prompt_version,))
    else:
        cursor.execute("SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1")
    
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def list_runs(db_path):
    """
    Lists all unique run executions, sorting by timestamp descending.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, timestamp, model_version, prompt_version, config_hash,
               AVG(final_score) as avg_score,
               SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as reject_count
        FROM runs
        GROUP BY run_id
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "run_id": r[0],
            "timestamp": r[1],
            "model_version": r[2],
            "prompt_version": r[3],
            "config_hash": r[4],
            "avg_score": r[5],
            "reject_count": r[6]
        }
        for r in rows
    ]
