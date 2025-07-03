import os
import csv
import json
from deepeval import evaluate
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.agents.query_retriever import AgenticRAG
from digital_assistant.logs.logger import SimpleLogger

from dotenv import load_dotenv
load_dotenv()

# ─── 0. Setup logging ───────────────────────────────────────────────────────
logger = SimpleLogger(__name__, level="debug")
logger.info("Starting DeepEval for RAG agent")
current_dir = os.path.dirname(os.path.abspath(__file__))

# ─── 1. Load test data from CSV ─────────────────────────────────────────────
qna_file_path = os.path.join(current_dir,"Sample_Q&A.csv")

test_data = []
with open(qna_file_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Expecting columns: question, expected_answer
        test_data.append({
            "query": row["query"],
            "expected_answer": row["expected_answer"]
        })

# ─── 2. Initialize your RAG agent ───────────────────────────────────────────
db = ChromaDB()
query_retriever = AgenticRAG(db)

def get_bot_answer(idx: int, query: str) -> str:
    """Calls your RAG pipeline and returns generated answer."""
    logger.debug(f"**********************Start - {idx}************************************")
    answer = query_retriever.run(query)
    logger.debug(f"**********************End - {idx}************************************")
    return answer

# ─── 3. Create DeepEval test cases ───────────────────────────────────────────
checkpoint_file = "checkpoint_outputs.json"
test_cases = []

if os.path.exists(checkpoint_file):
    logger.info(f"Found checkpoint file: {checkpoint_file}. Loading saved outputs...")
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved_outputs = json.load(f)
else:
    saved_outputs = []

# If checkpoint data exists and has correct number of test cases
if saved_outputs and len(saved_outputs) == len(test_data):
    for item in saved_outputs:
        test_cases.append(
            LLMTestCase(
                input=item["query"],
                actual_output=item["bot_answer"]["answer"],
                expected_output=item["expected"]
            )
        )
    logger.info("✅ Loaded all test cases from checkpoint.")
else:
    logger.info("⚡ No valid checkpoint found or test data changed. Generating outputs...")
    # Run pipeline and generate outputs
    saved_outputs = []
    for idx, item in enumerate(test_data):
        bot_answer = get_bot_answer(idx, item["query"])
        entry = {
            "query": item["query"],
            "bot_answer": bot_answer,
            "expected": item["expected_answer"]
        }
        saved_outputs.append(entry)

        test_cases.append(
            LLMTestCase(
                input=item["query"],
                actual_output=bot_answer,
                expected_output=item["expected_answer"]
            )
        )

    # Save checkpoint
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(saved_outputs, f, indent=2)
    logger.info(f"💾 Saved checkpoint to {checkpoint_file}.")

# ─── 4. Define evaluation metrics ──────────────────────────────────────────
metrics = [
    GEval(
        name="Correctness",
        criteria="Check if the actual output matches the expected output.",
        model="gpt-4o",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    ),
    AnswerRelevancyMetric(threshold=0.7)
]

# ─── 5. Run evaluation ─────────────────────────────────────────────────────
results = evaluate(test_cases, metrics)

# ─── 6. Prepare and save results ────────────────────────────────────────────

def flatten_metrics(metrics_data, prefix_keys=True, list_output=False):
    """
    Flatten a list of MetricData objects into one generic record/dict or flat list.

    Args:
        metrics_data: List of MetricData objects
        prefix_keys: If True, prefix each metric's fields with metric name or index
        list_output: If True, return a flat list in consistent key order; otherwise return dict

    Returns:
        dict or list
    """
    flat_dict = {}
    key_order = []

    for idx, metric in enumerate(metrics_data, start=1):
        base = metric.name.replace(" ", "_").replace("[","").replace("]","").replace(":", "")
        prefix = f"{base}" if prefix_keys else f"metric{idx}"

        for field in ['score', 'success', 'threshold', 'reason', 'evaluation_model', 'evaluation_cost']:
            key = f"{prefix}_{field}"
            flat_dict[key] = getattr(metric, field, None)
            key_order.append(key)

    if list_output:
        return [flat_dict[k] for k in key_order]
    return flat_dict


detailed = []
for tc, res in zip(test_cases, results.test_results):
    # Flatten all metrics into one record
    metrics_flat = flatten_metrics(res.metrics_data)

    record = {
        "query": tc.input,
        "bot_answer": tc.actual_output,
        "expected": tc.expected_output,
        **metrics_flat
    }
    detailed.append(record)

output = {
    "full_results": detailed
}

# File paths
json_file_path = os.path.join(current_dir, "deepeval_imdb_results.json")
csv_file_path = os.path.join(current_dir, "deepeval_imdb_results.csv")

# Save full results as JSON
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

# Save per-case results as CSV
fieldnames = list(detailed[0].keys()) if detailed else ["query", "bot_answer", "expected"]

with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for d in detailed:
        writer.writerow(d)

logger.info("✅ Evaluation complete. Results saved to deepeval_imdb_results.json and .csv")