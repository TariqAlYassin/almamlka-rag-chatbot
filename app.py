import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mamlka.answering import answer_question
from mamlka.observability import setup_observability
from mamlka.vector_store import get_qdrant_client

load_dotenv()
setup_observability()

st.set_page_config(
    page_title="Almamlaka TV | AI Assistant",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Almamlaka brand colors
BRAND_RED = "#98151C"
BRAND_DARK = "#720D13"
BRAND_LIGHT = "#F8F1F0"
BRAND_BORDER = "#E8D5D4"

# Custom styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #FCFAF9;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent;
    }}

    [data-testid="stSidebar"] {{
        background-color: {BRAND_LIGHT};
        border-right: 1px solid {BRAND_BORDER};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {BRAND_DARK};
    }}

    .brand-header {{
        background: linear-gradient(135deg, {BRAND_DARK}, {BRAND_RED});
        padding: 1.8rem 1.4rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 18px rgba(114, 13, 19, 0.14);
    }}

    .brand-header h1 {{
        color: white !important;
        font-size: 2rem;
        margin: 0.5rem 0 0.2rem 0;
        font-weight: 700;
    }}

    .brand-header p {{
        color: #F9E7E7 !important;
        margin: 0.45rem 0 0 0;
        font-size: 0.95rem;
    }}

    .brand-label {{
        color: #F9DCDD;
        font-size: 0.78rem;
        letter-spacing: 2px;
        font-weight: 600;
    }}

    [data-testid="stChatMessage"] {{
        border: 1px solid {BRAND_BORDER};
        border-radius: 14px;
        padding: 0.8rem;
        background-color: white;
    }}

    [data-testid="stChatInput"] {{
        border-color: {BRAND_RED};
        border-radius: 14px;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: {BRAND_DARK};
        box-shadow: 0 0 0 1px {BRAND_RED};
    }}

    .stButton > button {{
        background-color: {BRAND_RED};
        color: white;
        border: 1px solid {BRAND_RED};
        border-radius: 10px;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: {BRAND_DARK};
        color: white;
        border-color: {BRAND_DARK};
    }}

    .footer-note {{
        text-align: center;
        color: #817271;
        font-size: 0.78rem;
        margin-top: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# Load the uploaded Almamlaka logo, if present
logo_path = Path("assets/almamlaka_logo.jpg")
logo_html = ""

if logo_path.exists():
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    logo_html = (
        '<img src="data:image/png;base64,'
        + logo_base64
        + '" style="width:145px;max-width:70%;'
        'height:auto;object-fit:contain;" alt="Almamlaka TV logo">'
    )

# Branded header
st.markdown(
    f"""
    <div class="brand-header">
        {logo_html}
        <div class="brand-label">ALMAMLAKA TV</div>
        <h1>AI Knowledge Assistant</h1>
        <p>
            Ask questions about the supplied project documents.
            <br>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    if logo_html:
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem 0 1rem;">{logo_html}</div>',
            unsafe_allow_html=True,
        )

    st.title("About this assistant")
    st.write(
        "An AI assistant that answers questions using the "
        "provided project documents and cites its sources."
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Almamlaka TV | Project Knowledge Assistant")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reuse one Qdrant client during the session
if "qdrant_client" not in st.session_state:
    try:
        st.session_state.qdrant_client = get_qdrant_client()
    except Exception:
        st.error("Could not connect to the knowledge database. Please try again later.")
        st.stop()

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept a new question
question = st.chat_input("Ask about the project...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Almamlaka AI is preparing your answer..."):
            try:
                answer = answer_question(
                    question=question,
                    client=st.session_state.qdrant_client,
                    conversation_history=st.session_state.messages[:-1],
                )
            except Exception:
                answer = (
                    "Sorry, I couldn't generate an answer. "
                    "Please try again later or check the application "
                    "configuration."
                )
                st.error(
                    "The request failed. Check the application logs "
                    "for technical details."
                )

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="footer-note">'
    "Answers are based on the supplied documents and should include "
    "source citations.<br>"
    "الإجابات مستندة إلى الوثائق المقدمة."
    "</div>",
    unsafe_allow_html=True,
)
