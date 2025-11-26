import argparse
import json
import os
from agent.graph_hybrid import build_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Input JSONL file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    args = parser.parse_args()

    # Load Questions
    with open(args.batch, 'r') as f:
        questions = [json.loads(line) for line in f]

    # Build Graph
    app = build_graph()
    
    results = []
    
    print(f"Processing {len(questions)} questions...")
    
    for item in questions:
        print(f"Running: {item['id']}")
        
        initial_state = {
            "id": item['id'],
            "question": item['question'],
            "format_hint": item['format_hint'],
            "repair_attempts": 0,
            "error": None
        }
        
        # Increase recursion_limit just in case
        config = {"recursion_limit": 50}
        
        final_state = app.invoke(initial_state, config=config)
        
        output = {
            "id": item['id'],
            "final_answer": final_state.get("final_answer"),
            "sql": final_state.get("sql_query", ""),
            "confidence": 0.0,
            "explanation": final_state.get("explanation", ""),
            "citations": final_state.get("citations", [])
        }
        
        # Heuristic Confidence
        conf = 0.5
        if final_state.get("sql_rows"): conf += 0.3
        if final_state.get("retrieved_docs"): conf += 0.1
        if final_state.get("repair_attempts", 0) > 0: conf -= 0.2
        output["confidence"] = round(max(0.0, min(1.0, conf)), 2)

        print(f"Output: {output}")
        results.append(output)

    with open(args.out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"Done. Results written to {args.out}")

if __name__ == "__main__":
    main()