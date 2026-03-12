from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.rerank.router import classify_query

def classify_node(state: GraphState):
    state["query_type"] = classify_query(state["query"])
    return state

def retrieve_node(state: GraphState):
    # call bm25, dense, hybrid, muvera proxy here
    return state

def rerank_node(state: GraphState):
    # route to ColBERT / ColQwen2 based on query_type
    return state

def answer_node(state: GraphState):
    state["answer"] = "Stub answer"
    return state

graph = StateGraph(GraphState)
graph.add_node("classify", classify_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "answer")
graph.add_edge("answer", END)

app_graph = graph.compile()