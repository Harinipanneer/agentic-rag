import os
import json
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
import cohere

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode

from src.retrieval.fts_search import fts_search
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.vector_search import vector_search
from src.api.v1.agents.agent_utils import format_llm_output
from src.core.db import get_sql_database
from src.api.v1.schemas.query_schema import AIResponse

# =====================================
# ENV
# =====================================
load_dotenv(override=True)
os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"

# =====================================
# STATE + MODELS
# =====================================
class _RouteDecision(BaseModel):
    route: Literal["product", "document"]
    reason: str

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[dict]
    reranked_docs: List[dict]
    response: dict
    route: str
    generated_sql: str
    sql_result: str
    is_valid: bool
    attempts: int

# =====================================
# LLM
# =====================================
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =====================================
# GUARDRAIL NODE
# =====================================
def guardrail(query: str):
    print("\n========== GUARDRAIL CHECK ==========")

    # FIXED: Removed f-string, using correct variable passing to avoid parsing errors
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict document relevance classifier."),
        ("human", """Your job:
    Check if the user query is related to the uploaded documents.
         DON'T ANSWER OUT OF SCOPE

    Allow:
    - finance, HR, business, policies, reports only related to the uploaded documents
    - anything related to document content

    

    Reject:
    - random unrelated queries like food, entertainment, sports
    

    Query: {query}

    Answer ONLY YES or NO.""")
    ])

    res = (prompt | llm).invoke({"query": query})
    content = format_llm_output(res)
    print(f"[guardrail] Result: {content}")

    return "YES" not in content.upper()

# =====================================
# TOOLS
# =====================================
@tool
def vector_search_tool(query: str) -> list:
    """Use for semantic / natural language queries"""
    print("[tool] VECTOR TOOL USED")
    return vector_search(query, k=10)

@tool
def fts_search_tool(query: str) -> list:
    """Use for keyword / exact match queries"""
    print("[tool] FTS TOOL USED")
    return fts_search(query, k=10)

@tool
def hybrid_search_tool(query: str) -> list:
    """Use when query has both keyword + semantic meaning"""
    print("[tool] HYBRID TOOL USED")
    return hybrid_search(query, k=10)

tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]
llm_with_tools = llm.bind_tools(tools)

# =====================================
# RETRIEVE AGENT
# =====================================
def build_retrieve_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Retrieval Assistant.

1. Clean and normalize the query
2. Choose ONE appropriate tool
3. Call the tool

TOOLS:
- vector → natural language queries / conceptual questions
- fts → exact keywords, best for codes, IDs, abbreviations
- hybrid → short queries with both keyword + semantic meaning

RULES:
- Fix spelling and normalize terms
- Return ONE clean query only
- Call EXACTLY ONE tool
- Do NOT answer yourself
- Do NOT use outside knowledge"""),
        ("human", "{query}")
    ])
    return prompt | llm_with_tools

# =====================================
# ROUTER NODE
# =====================================
def router_node(state: RAGState) -> RAGState:
    print("\n========== ROUTER NODE ==========")
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router for an agentic RAG system. 
        Classify the user's query into EXACTLY one of two routes:

        "product" — If the query asks only about products, product-prices, stock/inventory, 
                    product categories, customer orders, or any query related to 
                    structured e-commerce data which is present in the database.

        "document" — the query asks about policies, procedures, guidelines, regulations, 
                     or any topic that requires reading text documents."""),
        ("human", "Query: {query}")
    ])

    decision = (prompt | structured_llm).invoke({"query": state["query"]})
    print(f"[router_node] Route: {decision.route}")
    print(f"[router_node] Reason: {decision.reason}")

    return {**state, "route": decision.route}

# =====================================
# NL2SQL NODE
# =====================================
def nl2sql_node(state: RAGState) -> RAGState:
    print("\n========== NL2SQL NODE ==========")

    db = get_sql_database()
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a PostgreSQL expert. Given the database schema below, 
        write a single valid SELECT query that answers the user's question.

        Rules:
        - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
        - Use only the tables and columns present in the schema.
        - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
        - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
        - For product or text searches: NEVER search for the full multi-word phrase as one
          ILIKE pattern. Instead, split the search into individual meaningful keywords
          and OR them together across both name and description columns.

        IMPORTANT:
        If the question asks for "most", "highest", "maximum", "top", or similar superlatives,
        and multiple rows may share the same highest value,
        DO NOT use ORDER BY ... LIMIT 1.
        Instead, return ALL rows that tie for the highest value by comparing against MAX(...).

        Database schema:
        {schema}"""
    ),
    ("human", "Question: {question}")
    ])

    raw_sql = (sql_prompt | llm).invoke({
        "schema": schema_info,
        "question": state["query"]
    })

    generated_sql = format_llm_output(raw_sql)
    generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    generated_sql = " ".join(generated_sql.split())
    print(f"[nl2sql_node] Generated SQL: {generated_sql}")

    try:
        sql_result = db.run(generated_sql)
    except Exception as exc:
        print(f"[nl2sql_node] SQL Error: {exc}")
        sql_result = f"SQL execution error: {exc}"

    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful data analyst. Answer the user's question using "
            "the SQL query results below. Be concise and format numbers/lists clearly. "
            "Set policy_citations to empty string, "
            "page_no to 'N/A', and document_name to 'agentic_rag_db'."
        ),
        (
            "human",
            "Question: {query}\n\n"
            "SQL Used:\n{sql}\n\n"
            "Query Results:\n{result}"
        )
    ])

    answer = (answer_prompt | structured_llm).invoke({
        "query": state["query"],
        "sql": generated_sql,
        "result": sql_result
    })

    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql

    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response
    }

# =====================================
# RETRIEVE NODE
# =====================================
def retrieve_node(state: RAGState) -> RAGState:
    print("\n========== RETRIEVE NODE ==========")

    agent = build_retrieve_agent()
    response = agent.invoke({"query": state["query"]})

    print(f"[retrieve] Tool calls detected: {bool(response.tool_calls)}")

    docs = []
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        name = tool_call["name"]
        query = tool_call["args"]["query"]
        print(f"[retrieve] Using Tool: {name} | Arg Query: {query}")

        if name == "fts_search_tool":
            docs = fts_search(query, 10)
        elif name == "vector_search_tool":
            docs = vector_search(query, 10)
        else:
            docs = hybrid_search(query, 10)
    else:
        # FIXED: Robust JSON fallback string parsing
        try:
            print("[retrieve] Falling back to text parsing...")
            if isinstance(response.content, list):
                text = response.content[0].get("text", "")
            else:
                text = str(response.content)
                
            parsed = json.loads(text.replace("```json", "").replace("```", "").strip())
            tool_name = parsed.get("tool", "")
            query_args = parsed.get("query", "")

            print(f"[retrieve] Parsed Fallback Tool: {tool_name} | Query: {query_args}")

            if tool_name == "fts":
                docs = fts_search(query_args, 10)
            elif tool_name == "vector":
                docs = vector_search(query_args, 10)
            else:
                docs = hybrid_search(query_args, 10)

        except Exception as e:
            print(f"[retrieve] Fallback parsing failed: {e}")
            docs = []

    print(f"[retrieve] Extracted {len(docs)} documents.")
    return {**state, "retrieved_docs": docs}

# =====================================
# RERANK NODE
# =====================================
def rerank_node(state: RAGState) -> RAGState:
    print("\n========== RERANK NODE ==========")
    if not state["retrieved_docs"]:
        print("[rerank] No documents to rerank.")
        return {**state, "reranked_docs": []}

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    
    # Safety: ensure we don't request more top_n than documents we have
    doc_contents = [d.get("content", "") for d in state["retrieved_docs"]]
    top_k = min(3, len(doc_contents))
    
    try:
        res = co.rerank(
            model="rerank-english-v3.0",
            query=state["query"],
            documents=doc_contents,
            top_n=top_k
        )
        reranked = [state["retrieved_docs"][r.index] for r in res.results]
        print(f"[rerank] Successfully reranked to top {len(reranked)} documents.")
    except Exception as e:
        print(f"[rerank] Error during reranking: {e}")
        reranked = state["retrieved_docs"][:top_k] # Fallback to raw top docs if API fails
        
    return {**state, "reranked_docs": reranked}

# =====================================
# VALIDATE NODE
# =====================================
def validate_node(state: RAGState) -> RAGState:
    print("\n========== VALIDATION NODE ==========")
    context = "\n\n".join(d.get("content", "") for d in state["reranked_docs"])

    # FIXED: Replaced f-string with variable passing to prevent LangChain parsing crashes
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict relevance checker. You must only output YES or NO."),
        ("human", """Task: Determine whether the CONTEXT is relevant to the QUESTION.

Rules:
- If the context discusses the query asked by the user  → YES
- If the context provides supporting  or partial information to the query → YES
- If the context is clearly unrelated → NO
- Do NOT require an explicit or complete answer
- Do NOT judge answer quality or completeness
- Do NOT refuse for missing details

QUESTION: {query}

CONTEXT:
{context}

Answer ONLY YES or NO.""")
    ])

    res = (prompt | llm).invoke({
        "query": state['query'],
        "context": context
    })
    
    output_content = format_llm_output(res).strip().upper()
    print(f"[validate] LLM Output: {output_content}")
    
    is_valid = "YES" in output_content
    print(f"[validate] Is Valid? {is_valid}")
    
    return {**state, "is_valid": is_valid}

# =====================================
# REWRITE NODE
# =====================================
def rewrite_node(state: RAGState) -> RAGState:
    print("\n========== REWRITE NODE ==========")
    
    # FIXED: Replaced f-string with proper variable injection
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful search query rewritter."),
        ("human", """Rewrite this query to a closely related or alternative form.
        Replace the main keyword with similar, broader, or related concepts from the same domain.
        Maintain the original intent while allowing substitution.
        Keep the query concise and retrieval-friendly.
        
        Return ONLY the rewritten query:
        
        {query}""")
    ])

    res = (prompt | llm).invoke({"query": state['query']})
    rewritten_query = format_llm_output(res).strip()
    
    attempts = state.get("attempts", 0) + 1
    print(f"[rewrite] Attempt {attempts} | Old Query: '{state['query']}' | New Query: '{rewritten_query}'")
    
    return {
        **state,
        "query": rewritten_query,
        "attempts": attempts
    }

# =====================================
# GENERATE NODE
# =====================================
def generate_node(state: RAGState) -> RAGState:
    print("\n========== GENERATE NODE ==========")

    llm_structured = llm.with_structured_output(AIResponse)

    # ✅ SQL PATH
    if state.get("generated_sql"):
        print("[generate] Using SQL path payload.")
        response = state["response"]
        response["sql_query_executed"] = state["generated_sql"]
        return {**state, "response": response}

    # ✅ DOCUMENT / RAG PATH
    docs = state.get("reranked_docs", [])

    context = "\n\n".join([
        f"[Source: {doc.get('source_file', 'unknown')} | "
        f"Page: {doc.get('page_number', 'N/A')} | "
        f"Section: {doc.get('section', 'N/A')}]\n"
        f"{doc.get('content', '')}"
        for doc in docs
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Financial Assistant.
            Answer the question in a natural, conversational sentence, clearly and concisely.
            Answer like you are explaining to a person.
            Combine relevant information from all provided context.
            Format the answer in a human-readable way.
            Do NOT include raw tables.
            Do NOT include unnecessary details.
            Keep it simple and clear.
            Include the relevant value naturally in the sentence."""
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {query}"
        )
    ])

    chain = prompt | llm_structured

    # FIXED: Removed the arbitrary [:3000] slice which was destroying your document context!
    result = chain.invoke({
        "context": context,
        "query": state["query"]
    })

    print("[generate] Answer generated successfully.")

    # ✅ METADATA EXTRACTION
    pages = sorted({
        str(doc.get("page_number"))
        for doc in docs
        if doc.get("page_number") is not None
    })

    sections = sorted({
        doc.get("section")
        for doc in docs
        if doc.get("section")
    })

    documents = sorted({
        doc.get("source_file")
        for doc in docs
        if doc.get("source_file")
    })

    response = result.model_dump()
    response["page_no"] = ", ".join(pages) if pages else None
    response["policy_citations"] = ", ".join(sections) if sections else None
    response["document_name"] = ", ".join(documents) if documents else None
    response["sql_query_executed"] = None  

    print(f"[generate] Pages: {pages}")
    print(f"[generate] Sections: {sections}")
    print(f"[generate] Documents: {documents}")

    return {
        **state,
        "response": response
    }

# =====================================
# GRAPH
# =====================================
def route_after_validate(state: RAGState) -> str:
    if state["is_valid"] or state["attempts"] >= 3:
        print("[router] Routing to -> GENERATE")
        return "generate"
    print("[router] Routing to -> REWRITE")
    return "rewrite"

def build_graph():
    g = StateGraph(RAGState)

    g.add_node("router", router_node)
    g.add_node("nl2sql", nl2sql_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("validate", validate_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"product": "nl2sql", "document": "retrieve"}
    )

    g.add_edge("nl2sql", END)
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "validate")

    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"generate": "generate", "rewrite": "rewrite"}
    )

    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile()

rag_app = build_graph()

# =====================================
# RUNNER
# =====================================
def run_rag_agent(query: str):
    print("\n========== AGENT START ==========")

    # ✅ GUARDRAIL SHORT-CIRCUIT
    if guardrail(query):
        print("[runner] Query blocked by guardrail.")
        return {
            "query": query,
            "answer": "Sorry I can help only with financial queries.",
            "policy_citations": None,
            "page_no": None,
            "document_name": None,
            "sql_query_executed": None,
        }

    state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "is_valid": False,
        "attempts": 0,
    }

    print(f"[runner] Invoking graph for query: '{query}'")
    final_state = rag_app.invoke(state)
    print("\n========== AGENT FINISHED ==========")
    return final_state["response"]