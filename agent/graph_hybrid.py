import dspy
import json
import re
import ast
from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END

# Import local modules
from agent.dspy_signatures import RouterSignature, GenerateSQL, SynthesizeAnswer
from agent.rag.retrieval import LocalRetriever
from agent.tools.sqlite_tool import SQLiteTool

# --- CONFIG ---
MODEL_NAME = "ollama/phi3.5:3.8b-mini-instruct-q4_K_M"

# FIX 1: Increase max tokens (num_predict) to prevent cutoff
lm = dspy.LM(model=MODEL_NAME, api_base="http://localhost:11434", num_predict=1000)
dspy.configure(lm=lm)

# Define State
class AgentState(TypedDict):
    id: str
    question: str
    format_hint: str
    strategy: str
    retrieved_docs: List[dict]
    extracted_constraints: str
    sql_query: str
    sql_columns: List[str]
    sql_rows: List[Any]
    error: Optional[str]
    repair_attempts: int
    final_answer: Any
    explanation: str
    citations: List[str]

# --- Helper: Robust SQL Extractor ---
def extract_sql_from_text(text: str) -> str:
    """Finds the first valid SQL block using regex, ignoring chatty intro text."""
    match = re.search(r"```(?:sql|sqlite)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    
    match = re.search(r"(SELECT\s+.*?(?:;|$))", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
        
    clean = text.strip()
    if clean.upper().startswith("SELECT") or clean.upper().startswith("WITH"):
        return clean
    
    return ""

# --- Nodes ---

def router_node(state: AgentState):
    print(f"   [1/6] Routing...")
    q_lower = state["question"].lower()
    if "policy" in q_lower or "return window" in q_lower:
        print("         > Strategy: rag (Heuristic)")
        return {"strategy": "rag"}
    print(f"         > Strategy: hybrid")
    return {"strategy": "hybrid"}

def retriever_node(state: AgentState):
    print(f"   [2/6] Retrieving docs...")
    retriever = LocalRetriever()
    docs = retriever.search(state["question"], k=6)
    
    if "policy" in state["question"].lower():
        policy_hits = retriever.search("product policy returns", k=3)
        existing_ids = {d['id'] for d in docs}
        for hit in policy_hits:
            if hit['id'] not in existing_ids:
                docs.append(hit)
                
    return {"retrieved_docs": docs}

def planner_node(state: AgentState):
    print(f"   [3/6] Planning...")
    docs_text = "\n\n".join([d['content'] for d in state["retrieved_docs"]])
    
    hints = """
    SQL Hints:
    1. Table "Order Details" MUST be quoted. Alias it as 'od'.
    2. JOIN PATH: Orders o JOIN "Order Details" od ON o.OrderID=od.OrderID JOIN Products p ON od.ProductID=p.ProductID.
    3. DATES: Use string format 'YYYY-MM-DD'.
    4. REVENUE: SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)).
    """
    constraints = f"Context:\n{docs_text}\n{hints}"
    return {"extracted_constraints": constraints}

def sql_gen_node(state: AgentState):
    current_attempt = state.get('repair_attempts', 0) + 1
    print(f"   [4/6] Generating SQL (Attempt {current_attempt})...")
    
    db = SQLiteTool()
    schema_str = db.get_schema()
    
    predictor = dspy.Predict(GenerateSQL)
    
    try:
        predictor.load("agent/optimized_sql_module.json")
    except:
        pass 
    
    response = predictor(
        db_schema=schema_str,
        constraints=state.get("extracted_constraints", ""),
        question=state["question"],
        previous_error=state.get("error") or "" 
    )
    
    raw_text = response.sql_query
    print(f"         > Raw SQL: {raw_text}")
    clean_sql = extract_sql_from_text(raw_text)
    clean_sql = clean_sql.replace("OrderDetails", '"Order Details"')
    
    print(f"         > Clean SQL: {clean_sql}...")
    
    return {"sql_query": clean_sql, "error": None}

def executor_node(state: AgentState):
    print(f"   [5/6] Executing SQL...")
    
    if not state["sql_query"]:
        print("         > No SQL generated.")
        return {
            "sql_rows": [], 
            "error": "No SQL generated", 
            "repair_attempts": state.get("repair_attempts", 0) + 1
        }

    db = SQLiteTool()
    cols, rows, err = db.execute_query(state["sql_query"])
    
    if err:
        print(f"         > SQL Error: {err}")
        return {
            "error": err, 
            "repair_attempts": state.get("repair_attempts", 0) + 1
        }
    
    print(f"         > Rows returned: {len(rows)}")
    return {"sql_columns": cols, "sql_rows": rows}

def synthesizer_node(state: AgentState):
    print(f"   [6/6] Synthesizing final answer...")
    doc_context = "\n".join([f"[{d['id']}] {d['content']}" for d in state["retrieved_docs"]])
    
    rows = state.get("sql_rows", [])
    sql_res_str = str(rows)[:2000] 
    
    predictor = dspy.Predict(SynthesizeAnswer)
    response = predictor(
        question=state["question"],
        format_hint=state["format_hint"],
        sql_query=state.get("sql_query", "N/A"),
        sql_result=sql_res_str,
        doc_context=doc_context
    )
    
    raw_ans = response.final_answer
    parsed_ans = raw_ans
    
    try:
        val_str = str(raw_ans).strip()
        if "{" in val_str or "[" in val_str:
            val_str = val_str.replace("```json", "").replace("```", "").strip()
            try:
                parsed_ans = json.loads(val_str.replace("'", '"'))
            except:
                parsed_ans = ast.literal_eval(val_str)
        elif state["format_hint"] == "int":
            nums = re.findall(r"-?\d+", val_str.replace(",", ""))
            if nums: parsed_ans = int(nums[0])
        elif state["format_hint"] == "float":
            nums = re.findall(r"-?\d+\.?\d*", val_str.replace(",", ""))
            if nums: parsed_ans = float(nums[0])
    except Exception as e:
        print(f"         > Parsing Error: {e}")
        pass 

    citations = []
    if state.get("sql_query"):
        for t in ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]:
            if t.lower() in state["sql_query"].lower() or f'"{t}"' in state["sql_query"]:
                citations.append(t)
    if state.get("retrieved_docs"):
        for doc in state["retrieved_docs"]:
            citations.append(doc['id'])

    return {
        "final_answer": parsed_ans,
        "explanation": response.explanation,
        "citations": citations
    }

# --- Edges ---

def check_repair(state: AgentState):
    if state.get("error"):
        attempts = state.get("repair_attempts", 0)
        if attempts <= 2:
            print(f"         > Retrying (Attempt {attempts}/2)")
            return "retry"
        else:
            print("         > Max retries reached. Moving to Synthesizer.")
            return "synthesize" 
    return "synthesize"

def check_strategy(state: AgentState):
    if state.get("strategy") == "rag":
        print("         > Skipping SQL (Strategy: RAG)")
        return "synthesize"
    return "generate_sql"

# --- Graph ---

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_gen", sql_gen_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "planner")
    workflow.add_conditional_edges("planner", check_strategy, {"synthesize": "synthesizer", "generate_sql": "sql_gen"})
    workflow.add_edge("sql_gen", "executor")
    workflow.add_conditional_edges("executor", check_repair, {"retry": "sql_gen", "synthesize": "synthesizer"})
    workflow.add_edge("synthesizer", END)
    return workflow.compile()