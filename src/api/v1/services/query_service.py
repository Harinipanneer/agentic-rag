
from src.api.v1.agents.agent import run_rag_agent


def run_agent(query: str):
    print("\n Running RAG Agent...")
    return run_rag_agent(query)

