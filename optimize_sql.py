import dspy
from dspy.teleprompt import BootstrapFewShot
from agent.dspy_signatures import GenerateSQL
from agent.tools.sqlite_tool import SQLiteTool

# NOTE: We use the full model tag here. 
# Run 'ollama list' in your terminal to verify this matches your downloaded model.
MODEL_NAME = "ollama/phi3.5:3.8b-mini-instruct-q4_K_M"

# Mock dataset for training (Tiny split as requested)
train_examples = [
    dspy.Example(
        db_schema="Table Products: ProductID (INTEGER), ProductName (TEXT), UnitPrice (FLOAT)...",
        constraints="CostOfGoods ≈ 0.7 * UnitPrice",
        question="What is the price of Chai?",
        previous_error="",
        sql_query="SELECT UnitPrice FROM Products WHERE ProductName = 'Chai'"
    ).with_inputs("db_schema", "constraints", "question", "previous_error"),
    dspy.Example(
        db_schema="Table Orders: OrderID (INTEGER), OrderDate (TEXT)...",
        constraints="Summer 1997 starts 1997-06-01",
        question="How many orders in Summer 1997?",
        previous_error="",
        sql_query="SELECT COUNT(*) FROM Orders WHERE OrderDate >= '1997-06-01' AND OrderDate <= '1997-06-30'"
    ).with_inputs("db_schema", "constraints", "question", "previous_error")
]

def optimize_sql_module():
    # Setup
    lm = dspy.LM(model=MODEL_NAME, api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    
    # Metric: check if SQL is valid string and starts with SELECT
    def validate_sql(example, pred, trace=None):
        sql = pred.sql_query.strip().upper()
        # Basic validation: starts with SELECT/WITH and is not empty
        return (sql.startswith("SELECT") or sql.startswith("WITH")) and len(sql) > 10

    # Optimizer
    print(f"Optimizing NL->SQL module using {MODEL_NAME}...")
    # max_bootstrapped_demos=2 means it will try to generate 2 refined examples
    teleprompter = BootstrapFewShot(metric=validate_sql, max_bootstrapped_demos=2)
    
    # Compile
    student = dspy.ChainOfThought(GenerateSQL)
    
    try:
        optimized_program = teleprompter.compile(student, trainset=train_examples)
        
        # Save
        optimized_program.save("agent/optimized_sql_module.json")
        print("Optimization complete. Saved to agent/optimized_sql_module.json")
        
    except Exception as e:
        print(f"\nOptimization Failed. Error details:\n{e}")
        print("\nTip: Check if 'ollama list' shows the exact model name used in the script.")

if __name__ == "__main__":
    optimize_sql_module()