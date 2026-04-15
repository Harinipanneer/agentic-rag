import os
import json
import ast
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Annotated
from dotenv import load_dotenv
import cohere

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.runnables.graph import MermaidDrawMethod

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
    route: Literal["product", "document", "both"]
    reason: str

class RAGState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
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
    print(f"\n🛡️  [GUARDRAIL] Checking: '{query}'")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the NorthStar Bank Domain Guardrail."),
        ("human", """Your job: 
        Determine if the user query is related to NorthStar Bank Credit Cards, 
        financial transactions, reward programs, or bank policies.

        ALLOW: Finance, card variants, spending habits, fees, and billing.
        REJECT: General chat, sports, movies, or non-banking topics.

        Query: {query}
        Answer ONLY YES or NO.""")
    ])

    res = (prompt | llm).invoke({"query": query})
    content = format_llm_output(res).strip().upper()
    is_blocked = "YES" not in content
    print(f"🛡️  [GUARDRAIL] Result: {'🚫 BLOCKED' if is_blocked else '✅ PASSED'}")
    return is_blocked

# =====================================
# TOOLS
# =====================================
@tool
def vector_search_tool(query: str) -> list:
    """Semantic search for NorthStar policies and card benefits."""
    print(f"🛠️  [TOOL] Vector Search: '{query}'")
    return vector_search(query, k=20)

@tool
def fts_search_tool(query: str) -> list:
    """Exact keyword search for variants like 'Gold' or 'Signature'."""
    print(f"🛠️  [TOOL] Keyword Search: '{query}'")
    return fts_search(query, k=20)

@tool
def hybrid_search_tool(query: str) -> list:
    """Combined search for complex banking queries."""
    print(f"🛠️  [TOOL] Hybrid Search: '{query}'")
    return hybrid_search(query, k=20)

tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]
llm_with_tools = llm.bind_tools(tools)
retrieve_tools_node = ToolNode(tools)

# =====================================
# ROUTER NODE
# =====================================
def router_node(state: RAGState) -> RAGState:
    print("\n🚦 [ROUTER] Calculating optimal pathway...")
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Query Router for the NorthStar Credit Card Summarizer. 
        Classify the query into EXACTLY one of three routes:

        - "product": Queries about user-specific data (spending, limits, merchants).
        - "document": Queries about general banking policies (rewards, fees, lounge access).
        - "both": Queries comparing user spend against bank rules (e.g., fee waivers).
        """),
        ("human", "Query: {query}")
    ])

    decision = (prompt | structured_llm).invoke({"query": state["query"]})
    print(f"🚦 [ROUTER] Pathway: {decision.route.upper()}")
    print(f"🚦 [ROUTER] Reason: {decision.reason}")
    return {**state, "route": decision.route}

# =====================================
# NL2SQL NODE
# =====================================
def nl2sql_node(state: RAGState) -> RAGState:
    print("\n🗄️  [SQL] Synthesizing PostgreSQL query...")
    db = get_sql_database()
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a NorthStar Bank PostgreSQL Expert. 
    Write a SELECT query to answer the banking question using ONLY the provided schema.
    
    CRITICAL DOMAIN RULES:
    1. 'Spend' refers only to txn_type = 'purchase'. 
    2. 'Net Spend' = (Sum of 'purchase') - (Sum of 'refund').
    3. TIES & MAXIMUMS: Use subqueries/CTEs with MAX() to return ALL rows that tie for the highest value.
    4. For date ranges (e.g., 'March 2026'), refer to billing cycles if available, otherwise use calendar dates.
    5. Return ONLY the raw SQL. No markdown formatting.
    
    Database schema:
    {schema}"""),
    ("human", "Question: {question}")
    ])

    raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": state["query"]})
    generated_sql = format_llm_output(raw_sql).replace("```sql", "").replace("```", "").strip()
    
    print(f"🗄️  [SQL] Generated Query: {generated_sql}")
    
    try:
        sql_result = db.run(generated_sql)
        print(f"🗄️  [SQL] Result: {sql_result}")
    except Exception as exc:
        print(f"❌ [SQL] ERROR: {exc}")
        sql_result = f"SQL execution error: {exc}"

    return {**state, "generated_sql": generated_sql, "sql_result": str(sql_result)}

# =====================================
# RETRIEVE NODE
# =====================================
def retrieve_node(state: RAGState) -> dict:
    print("\n🔍 [RETRIEVAL] Selecting best PDF search tool...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a NorthStar Product Guide Assistant. Select a tool to find 
        card features, reward multipliers, or fee schedules."""),
        ("human", "{query}")
    ])
    agent = prompt | llm_with_tools
    response = agent.invoke({"query": state["query"]})
    
    if response.tool_calls:
        print(f"🛠️  [RETRIEVAL] Calling tool: {response.tool_calls[0]['name']}")
    return {"messages": [response]}

def should_continue_retrieval(state: RAGState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls: return "tools"
    print("➡️  [FLOW] Retrieval done. Proceeding to Rerank.")
    return "rerank"

# =====================================
# RERANK NODE
# =====================================
def rerank_node(state: RAGState) -> RAGState:
    print("\n🎯 [RERANK] Evaluating chunk relevance via Cohere English-v3.0...")
    docs = state.get("retrieved_docs", [])
    if not docs: 
        print("⚠️  [RERANK] No context chunks to refine.")
        return {**state, "reranked_docs": []}

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    doc_contents = [d.get("content", "") for d in docs]
    
    try:
        res = co.rerank(model="rerank-english-v3.0", query=state["query"], documents=doc_contents, top_n=min(10, len(docs)))
        reranked = [docs[r.index] for r in res.results]
        print(f"🎯 [RERANK] Optimized down to {len(reranked)} relevant chunks.")
    except Exception as e:
        print(f"⚠️  [RERANK] Falling back to standard order. Error: {e}")
        reranked = docs[:10]
        
    return {**state, "retrieved_docs": docs, "reranked_docs": reranked}

# =====================================
# HYBRID NODE (DOMAIN SYNTHESIS)
# =====================================
def hybrid_node(state: RAGState) -> RAGState:
    print("\n🏗️  [HYBRID] Synchronizing Database data with Policy context...")
    
    sql_state = nl2sql_node(state)
    state.update(sql_state)
    
    print("🏗️  [HYBRID] Retrieving supporting Policy documents...")
    docs = hybrid_search(state["query"], k=20)
    state["retrieved_docs"] = docs
    
    rerank_state = rerank_node(state)
    state.update(rerank_state)
    
    llm_structured = llm.with_structured_output(AIResponse)

    context_parts = []
    if state.get("sql_result"):
        context_parts.append(f"--- USER DATA ---\n{state['sql_result']}")
    if state.get("reranked_docs"):
        doc_text = "\n\n".join([f"[Source: {d.get('source_file')} | Page: {d.get('page_number')}]\n{d.get('content')}" for d in state["reranked_docs"]])
        context_parts.append(f"--- BANK POLICY ---\n{doc_text}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the NorthStar Bank Financial Assistant. 
        Combine Database Results and PDF Policy context to answer precisely.
        
        DOMAIN RULES:
        1. Professional Tone: No phrases like "According to the database". 
        2. Citing: You MUST accurately cite page_no and document_name only for chunks used.
        3. Financial Interpretation: Apply reward multipliers or fee logic found in the guide to the spend data provided.
        """),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    result = (prompt | llm_structured).invoke({"context": "\n\n".join(context_parts), "query": state["query"]})
    print("✅ [HYBRID] Multi-path answer generated.")
    
    response = result.model_dump()
    response["sql_query_executed"] = state.get("generated_sql")
    response["source_chunks"] = [f"[Page: {d.get('page_number', 'N/A')} | File: {d.get('source_file', 'N/A')}]\n{d.get('content')}" for d in state.get("reranked_docs", [])]

    return {**state, "response": response}

# =====================================
# VALIDATE & REWRITE NODES
# =====================================
def validate_node(state: RAGState) -> RAGState:
    print("\n⚖️  [VALIDATE] Running relevance audit...")
    context = "\n\n".join(d.get("content", "") for d in state["reranked_docs"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a relevance checker. You must only output YES or NO."),
        ("human", """Task: Determine whether the CONTEXT is relevant to the QUESTION.

    Rules:
    If the context discusses the query asked by the user  → YES
    If the context provides supporting  or partial information to the query → YES
    If the context is clearly unrelated → NO
    Do NOT require an explicit or complete answer
    Do NOT judge answer quality or completeness
    Do NOT refuse for missing details

QUESTION: {query}
CONTEXT:
{context}

Answer ONLY YES or NO.""")
    ])

    res = (prompt | llm).invoke({"query": state['query'], "context": context})
    is_valid = "YES" in format_llm_output(res).strip().upper()
    print(f"⚖️  [VALIDATE] Context valid? {'✅ YES' if is_valid else '❌ NO'}")
    return {**state, "is_valid": is_valid}

def rewrite_node(state: RAGState) -> RAGState:
    print("\n🔄 [REWRITE] Enhancing query for NorthStar Guide terminology...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a NorthStar Bank Query Optimizer."),
        ("human", "Rewrite the following query to use banking terms from the Product Guide. Query: {query}")
    ])

    res = (prompt | llm).invoke({"query": state['query']})
    rewritten = format_llm_output(res).strip()
    print(f"🔄 [REWRITE] Rewritten query: '{rewritten}'")
    return {**state, "query": rewritten, "attempts": state.get("attempts", 0) + 1}

# =====================================
# GENERATE NODE
# =====================================
def generate_node(state: RAGState) -> RAGState:
    print("\n✍️  [GENERATE] Preparing final user response...")
    llm_structured = llm.with_structured_output(AIResponse)

    context_parts = []
    if state.get("sql_result"): context_parts.append(f"--- TRANSACTION DATA ---\n{state['sql_result']}")
    if state.get("reranked_docs"):
        doc_text = "\n\n".join([f"[Source: {d.get('source_file')} | Page: {d.get('page_number')}]\n{d.get('content')}" for d in state["reranked_docs"]])
        context_parts.append(f"--- POLICY GUIDE ---\n{doc_text}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the NorthStar Bank Financial Assistant.
        Format the answer naturally and clearly. 
        Cite sources correctly using the structured fields provided.
        Example: "Your total spend for March 2026 is Rs. 42,300."
        """),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    result = (prompt | llm_structured).invoke({"context": "\n\n".join(context_parts), "query": state["query"]})
    print("✅ [GENERATE] Response ready.")
    
    response = result.model_dump()
    response["sql_query_executed"] = state.get("generated_sql")
    response["source_chunks"] = [f"[Page: {d.get('page_number', 'N/A')} | File: {d.get('source_file', 'N/A')}]\n{d.get('content')}" for d in state.get("reranked_docs", [])]

    return {**state, "response": response}

# =====================================
# GRAPH WIRING
# =====================================
def route_after_validate(state: RAGState) -> str:
    if state["is_valid"] or state["attempts"] >= 3:
        return "generate"
    print("🔁 [FLOW] Context insufficient. Retrying path...")
    return "rewrite"

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("router", router_node)
    g.add_node("nl2sql", nl2sql_node)
    g.add_node("hybrid", hybrid_node) 
    g.add_node("retrieve", retrieve_node)
    g.add_node("tools", retrieve_tools_node)
    g.add_node("rerank", rerank_node)
    g.add_node("validate", validate_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"], {"product": "nl2sql", "document": "retrieve", "both": "hybrid"})
    g.add_edge("nl2sql", "generate") 
    g.add_edge("hybrid", END) 
    g.add_conditional_edges("retrieve", should_continue_retrieval, {"tools": "tools", "rerank": "rerank"})
    g.add_edge("tools", "rerank")
    g.add_edge("rerank", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    
    return g.compile()

rag_app = build_graph()

def run_rag_agent(query: str):
    print(f"\n{'='*60}\n🏦 [SESSION START] Processing NorthStar Query\n{'='*60}")
    if guardrail(query):
        return {"query": query, "answer": "I can only assist with NorthStar Bank credit card queries."}

    state = {
        "query": query, "messages": [("user", query)], 
        "retrieved_docs": [], "reranked_docs": [], "response": {}, 
        "route": "", "generated_sql": "", "sql_result": "", "is_valid": False, "attempts": 0,
    }

    final_state = rag_app.invoke(state)
    print(f"\n{'='*60}\n🏁 [SESSION END] Request Fulfilled.\n{'='*60}")
    return final_state["response"]