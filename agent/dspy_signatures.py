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

    return a valid full complete SQLite query, do not add any extra text.
    """
    db_schema = dspy.InputField(desc="Schema with columns") 
    constraints = dspy.InputField(desc="Context & required date filters")
    question = dspy.InputField()
    previous_error = dspy.InputField(desc="Correction needed")
    sql_query = dspy.OutputField(desc="The SQL query starting with SELECT")

# 3. Synthesizer
class SynthesizeAnswer(dspy.Signature):
    """Answer the question based on SQL results and retrieved context."""
    question = dspy.InputField()
    format_hint = dspy.InputField()
    sql_query = dspy.InputField()
    sql_result = dspy.InputField(desc="Rows returned from database")
    doc_context = dspy.InputField(desc="Relevant text from documents")
    
    final_answer = dspy.OutputField(desc="The answer matching format_hint exactly")
    explanation = dspy.OutputField(desc="Brief explanation")
