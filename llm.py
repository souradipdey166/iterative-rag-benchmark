"""
Thin wrapper around the LLM call. Everything else in this project calls
call_llm() -- it doesn't know or care whether that's Groq, Claude, or
anything else. Swapping providers later means changing only this file.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


def call_llm(prompt: str) -> str:
    result = _llm.invoke(prompt)
    return result.content