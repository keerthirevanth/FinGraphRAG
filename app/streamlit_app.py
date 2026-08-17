"""Streamlit interface: vector RAG versus GraphRAG, side by side.

The app lets a user ask a question and see both systems answer it at once,
with the graph system's retrieved facts and the vector system's retrieved
chunks shown beneath each answer. A second tab renders the local neighbourhood
of the entities in the question as an interactive graph, which makes the
multi-hop connections the project is about visible at a glance. A third tab
shows the evaluation results if they have been generated.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.rag.vector_rag import VectorRAG
from src.rag.graph_rag import GraphRAG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "eval" / "results.json"

EXAMPLE_QUESTIONS = [
    "How is Tesla connected to TSMC?",
    "Which companies have a business relationship with Samsung?",
    "Who are the main competitors of NVIDIA?",
    "How is Microsoft connected to Micron Technology?",
    "Which corpus companies are exposed to Huawei?",
]


# Loading the models and graph is expensive, so cache them across reruns.
@st.cache_resource
def load_systems():
    return VectorRAG(), GraphRAG()


def render_answers(question: str) -> None:
    """Answer the question with both systems and show results side by side."""
    vector_rag, graph_rag = load_systems()
    left, right = st.columns(2)

    with left:
        st.subheader("Vector RAG")
        st.caption("Retrieves the most similar filing passages, then answers.")
        with st.spinner("Retrieving and answering..."):
            result = vector_rag.answer(question)
        st.markdown(result["answer"])
        with st.expander(f"Retrieved passages ({len(result['chunks'])})"):
            for chunk in result["chunks"]:
                st.markdown(
                    f"**{chunk['ticker']} - {chunk['section']}** "
                    f"(similarity {chunk['score']:.2f})"
                )
                st.caption(chunk["text"][:400] + "...")

    with right:
        st.subheader("GraphRAG")
        st.caption("Traverses the knowledge graph, then answers from the facts.")
        with st.spinner("Traversing and answering..."):
            result = graph_rag.answer(question)
        st.markdown(result["answer"])
        st.caption("Linked entities: " + (", ".join(result["entities"]) or "none"))
        with st.expander(f"Graph facts used ({len(result['facts'])})"):
            for fact in result["facts"]:
                st.markdown(f"- {fact}")


def render_graph_view(question: str) -> None:
    """Draw the neighbourhood of the question's entities as a network."""
    _, graph_rag = load_systems()
    entities = graph_rag.link_entities(question)
    if not entities:
        st.info("No known companies were found in the question to plot.")
        return

    graph = graph_rag.graph
    undirected = graph.to_undirected(as_view=True)

    # Collect the entities plus their immediate neighbours.
    nodes = set(entities)
    for entity in entities:
        nodes.update(undirected.neighbors(entity))

    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("Install pyvis to see the interactive graph: pip install pyvis")
        return

    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff")
    for node in nodes:
        is_corpus = graph.nodes[node].get("in_corpus", False)
        is_focus = node in entities
        color = "#e63946" if is_focus else ("#457b9d" if is_corpus else "#a8a8a8")
        size = 28 if is_focus else (20 if is_corpus else 14)
        net.add_node(node, label=node, color=color, size=size)

    for source, target, data in graph.edges(data=True):
        if source in nodes and target in nodes:
            label = ", ".join(data.get("relations", []))
            net.add_edge(source, target, title=label, label=label)

    net.set_options('{"physics": {"barnesHut": {"springLength": 160}}}')
    html = net.generate_html()
    st.components.v1.html(html, height=620)


def render_evaluation() -> None:
    """Show the evaluation summary if results exist."""
    if not RESULTS_FILE.exists():
        st.info("No evaluation results yet. Run: python -m src.eval.run_eval")
        return
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    types = ["simple", "multi_hop", "global", "ALL"]
    metrics = ["entity_recall", "correctness", "faithfulness"]
    systems = ["vector", "graph"]

    def average(system, qtype, metric):
        vals = [
            r[metric] for r in results
            if r["system"] == system
            and (qtype == "ALL" or r["type"] == qtype)
            and r[metric] is not None
        ]
        return sum(vals) / len(vals) if vals else None

    for metric in metrics:
        st.subheader(metric.replace("_", " ").title())
        rows = []
        for qtype in types:
            row = {"question type": qtype}
            for system in systems:
                value = average(system, qtype, metric)
                row[system] = None if value is None else round(value, 2)
            rows.append(row)
        st.table(rows)


def main() -> None:
    st.set_page_config(page_title="FinGraphRAG", layout="wide")
    st.title("FinGraphRAG: Knowledge-Graph RAG over Company Annual Filings")
    st.write(
        "Comparing graph-based retrieval against vector retrieval on questions "
        "about the AI / semiconductor / cloud ecosystem."
    )

    question = st.text_input(
        "Ask a question about the companies and their relationships:",
        value=EXAMPLE_QUESTIONS[0],
    )
    st.caption("Try: " + "  |  ".join(EXAMPLE_QUESTIONS[1:]))

    answers_tab, graph_tab, eval_tab = st.tabs(
        ["Answers", "Graph view", "Evaluation"]
    )
    with answers_tab:
        if question:
            render_answers(question)
    with graph_tab:
        if question:
            render_graph_view(question)
    with eval_tab:
        render_evaluation()


if __name__ == "__main__":
    main()
