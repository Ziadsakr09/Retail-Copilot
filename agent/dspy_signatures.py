import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """Classify user question into RAG, SQL, or Hybrid strategy."""
    question = dspy.InputField()
    strategy = dspy.OutputField(desc="One of: rag, sql, hybrid")

# 2. SQL Generator
class GenerateSQL(dspy.Signature):
    """You are a SQLite expert. Generate executable SQL queries.
    
    CRITICAL RULES:
    1. Table names: Orders, "Order Details" (MUST USE QUOTES), Products, Customers, Categories.
    2. JOIN PATH for Product Sales: Categories -> Products -> "Order Details" -> Orders.
    3. Dates: Use string format 'YYYY-MM-DD'.
    4. Output ONLY the SQL string. No markdown, no comments.
    """
    db_schema = dspy.InputField(desc="Schema with columns") 
    constraints = dspy.InputField(desc="Context & required date filters")
    question = dspy.InputField()
    previous_error = dspy.InputField(desc="Correction needed")
    sql_query = dspy.OutputField(desc="The SQL query starting with SELECT")

# 3. Synthesizer (UPDATED)
class SynthesizeAnswer(dspy.Signature):
    """Answer the question based on SQL results and retrieved context.
    
    Output Format:
    Answer: <The exact answer matching the format hint>
    Explanation: <A brief explanation>
    """
    question = dspy.InputField()
    format_hint = dspy.InputField()
    sql_query = dspy.InputField()
    sql_result = dspy.InputField(desc="Rows returned from database")
    doc_context = dspy.InputField(desc="Relevant text from documents")
    
    # Changed to single text field to prevent JSON parsing errors
    response_text = dspy.OutputField(desc="The answer and explanation text")
