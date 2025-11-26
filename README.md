# Retail Analytics Copilot (DSPy + LangGraph)

A local, private AI agent that answers retail analytics questions by orchestrating RAG over local documents and SQL generation against a SQLite database.

## Graph Design
The agent is implemented as a stateful **LangGraph** workflow with the following architecture:
*   **Hybrid Routing:** A `Router` node classifies questions to use either a pure RAG strategy (for policy/text questions) or a Hybrid SQL strategy (for data/analytics), preventing SQL hallucinations on text queries.
*   **Retrieval & Planning:** A `Retriever` fetches context from local Markdown docs, followed by a `Planner` that injects schema hints and extracts constraints (e.g., "Summer 1997" → `1997-06-01`).
*   **Self-Correcting SQL Loop:** The `SQL Generator` produces queries which are run by the `Executor`. If execution fails (e.g., syntax error), the error is fed back to the Generator for repair (up to 2 retries).
*   **Strict Synthesis:** A final `Synthesizer` combines SQL rows and document chunks to produce the exact requested JSON/Type format.

## DSPy Optimization
*   **Module Optimized:** `GenerateSQL` (Natural Language → SQLite).
*   **Optimizer:** `dspy.BootstrapFewShot` with a validity metric (checks for valid `SELECT` syntax).
*   **Impact (Valid SQL Rate):**
    *   **Before (Zero-Shot):** ~40%. The local model frequently hallucinated table names (e.g., missing quotes on `"Order Details"`) or failed complex joins.
    *   **After (Few-Shot):** ~85%. The optimizer successfully "taught" the model to strictly quote table names and follow the specific join paths defined in the training examples.

## Trade-offs & Assumptions
*   **Cost Approximation:** The Northwind database lacks a cost column. `CostOfGoods` is approximated as **0.7 * UnitPrice** as defined in the KPI documentation.
*   **Local Model Constraints:** To prevent "incomplete input" errors and timeouts on the local `Phi-3.5` (CPU) model, `ChainOfThought` was replaced with `dspy.Predict` for the SQL module, and token limits were explicitly increased.
*   **Date Interpretation:** "Summer Beverages 1997" is strictly filtered as **June 1997** (`1997-06-01` to `1997-06-30`) based on the specific entry in the `docs/marketing_calendar.md` file.
