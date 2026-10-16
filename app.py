from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mamlka.answering import answer_question
from mamlka.observability import setup_observability
from mamlka.vector_store import get_qdrant_client

setup_observability()
load_dotenv()


st.set_page_config(
    page_title="Almamlaka AI Assistant",
    page_icon="📺",
    layout="centered",
)

st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <h1 style="margin-bottom:0;">Almamlaka</h1>
        <h3 style="font-weight:400; margin-top:0.4rem;">
            AI Project Knowledge Assistant
        </h3>
        <p style="color:gray;">
            Ask questions about the project documents in Arabic or English.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("About")
    st.write(
        "This assistant answers questions using the supplied "
        "project documents and provides source citations."
    )
    st.caption("Powered by retrieval-augmented generation (RAG).")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.get("messages"):
    st.session_state.messages = []

# Reuse one local Qdrant client during the Streamlit session.
if "qdrant_client" not in st.session_state:
    try:
        st.session_state.qdrant_client = get_qdrant_client()
    except Exception as exc:
        st.error(f"Could not connect to the knowledge database: {exc}")
        st.stop()

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question.
question = st.chat_input("Ask about the project... / اسأل عن المشروع...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            try:
                answer = answer_question(
                    question=question,
                    client=st.session_state.qdrant_client,
                    conversation_history=st.session_state.messages[:-1],
                )
            except Exception as exc:
                answer = (
                    "Sorry, I couldn't generate an answer. "
                    "Please check the API configuration and database, "
                    "then try again."
                )
                st.error(f"Error details: {exc}")

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
