# app.py -- Gradio, deployed to Render / Hugging Face Spaces

import os

import gradio as gr
from datasets import load_dataset

from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot, answer_iterative


ds = load_dataset(
    "hotpot_qa",
    "distractor",
    split="validation[:50]"
)

question_choices = {
    ex["question"]: ex
    for ex in ds
}


def run_comparison(question_text):
    example = question_choices[question_text]

    paragraphs = [
        {
            "id": t,
            "title": t,
            "text": " ".join(s)
        }
        for t, s in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        )
    ]

    chunks = chunk_paragraphs(
        paragraphs,
        chunk_size=300,
        overlap=50
    )

    retriever = Retriever(collection_name="demo")
    retriever.index(chunks)

    single = answer_single_shot(
        question_text,
        retriever,
        top_k=5
    )

    iterative = answer_iterative(
        question_text,
        retriever,
        top_k=5,
        max_hops=3
    )

    return (
        f"{single['answer']}\n\n"
        f"Sources: {single['retrieved_source_ids']}",

        f"{iterative['answer']}\n\n"
        f"Hops: {iterative['hops']}\n"
        f"Sources: {iterative['retrieved_source_ids']}",

        example["answer"],
    )


demo = gr.Interface(
    fn=run_comparison,

    inputs=gr.Dropdown(
        choices=list(question_choices.keys()),
        label="Pick a HotpotQA question"
    ),

    outputs=[
        gr.Textbox(label="Single-shot answer"),
        gr.Textbox(label="Iterative (multi-hop) answer"),
        gr.Textbox(label="Gold answer"),
    ],

    title="Multi-hop RAG: Single-shot vs. Iterative Retrieval",

    description=(
        "Pick a real HotpotQA question and watch both strategies "
        "attempt it live. See results/manual_notes.md in the repo "
        "for the full 50-question ablation."
    ),
)


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=True
)