import json
import streamlit as st

from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot, answer_iterative


# -----------------------------
# Load demo dataset
# -----------------------------

with open("hotpotqa_50.json", "r", encoding="utf-8") as f:
    ds = json.load(f)

question_choices = {
    ex["question"]: ex
    for ex in ds
}


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Multi-hop RAG Benchmark",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-hop RAG: Single-shot vs Iterative Retrieval")

st.markdown(
    """
Compare two RAG strategies on real HotpotQA questions.

- **Single-shot RAG** → retrieves context once and answers.
- **Iterative RAG** → retrieves context, reasons about the result,
  generates another search query, and continues for multiple hops.
"""
)


# -----------------------------
# Question selector
# -----------------------------

question = st.selectbox(
    "Choose a HotpotQA question:",
    list(question_choices.keys())
)


# -----------------------------
# Run comparison
# -----------------------------

if st.button("🚀 Run RAG Comparison", type="primary"):

    example = question_choices[question]

    paragraphs = [
        {
            "id": title,
            "title": title,
            "text": " ".join(sentences)
        }
        for title, sentences in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        )
    ]

    with st.spinner("Running RAG agents..."):

        # Chunk documents
        chunks = chunk_paragraphs(
            paragraphs,
            chunk_size=300,
            overlap=50
        )

        # Create temporary Chroma collection
        retriever = Retriever(
            collection_name="demo"
        )

        retriever.index(chunks)

        # -------------------------
        # Single-shot RAG
        # -------------------------

        single = answer_single_shot(
            question,
            retriever,
            top_k=5
        )

        # -------------------------
        # Iterative RAG
        # -------------------------

        iterative = answer_iterative(
            question,
            retriever,
            top_k=5,
            max_hops=3
        )

    # -----------------------------
    # Results
    # -----------------------------

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Single-shot RAG")

        st.markdown("**Answer**")

        st.info(single["answer"])

        st.markdown("**Retrieved sources**")

        st.write(single["retrieved_source_ids"])

    with col2:

        st.subheader("Iterative RAG")

        st.markdown("**Answer**")

        st.info(iterative["answer"])

        st.markdown("**Hops used**")

        st.write(iterative["hops"])

        st.markdown("**Retrieved sources**")

        st.write(iterative["retrieved_source_ids"])

    # -----------------------------
    # Gold answer
    # -----------------------------

    st.divider()

    st.subheader("🎯 Gold Answer")

    st.success(example["answer"])

    st.caption(
        "Ground-truth answer from the HotpotQA example."
    )