import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mamlka.answering import answer_question
from mamlka.observability import setup_observability
from mamlka.vector_store import get_qdrant_client

# --------------------------------------------------
# Configuration
# --------------------------------------------------

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
PAGE_BACKGROUND = "#FCFAF9"
TEXT_COLOR = "#262020"


# --------------------------------------------------
# Logo
# --------------------------------------------------

logo_path = Path(__file__).parent / "assets" / "almamlaka_logo.jpg"
logo_html = ""

if logo_path.exists():
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    logo_html = (
        f'<img src="data:image/jpeg;base64,{logo_base64}" '
        'style="width:145px;max-width:70%;height:auto;'
        'object-fit:contain;" alt="Almamlaka TV logo">'
    )


# --------------------------------------------------
# Custom UI styling
# --------------------------------------------------

st.markdown(
    f"""
    <style>
    /* Main page: fixed light colors */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background-color: {PAGE_BACKGROUND} !important;
        color: {TEXT_COLOR} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    /* General text */
    .stApp p,
    .stApp li,
    .stApp label,
    [data-testid="stMarkdownContainer"] {{
        color: {TEXT_COLOR};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BRAND_LIGHT} !important;
        border-right: 1px solid {BRAND_BORDER};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {BRAND_DARK} !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {TEXT_COLOR} !important;
    }}

    /* Branded header */
    .brand-header {{
        background: linear-gradient(
            135deg,
            {BRAND_DARK},
            {BRAND_RED}
        );
        padding: 1.8rem 1.4rem;
        border-radius: 18px;
        text-align: center;
        color: #FFFFFF !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 18px rgba(114, 13, 19, 0.14);
    }}

    .brand-header h1 {{
        color: #FFFFFF !important;
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
        color: #F9DCDD !important;
        font-size: 0.78rem;
        letter-spacing: 2px;
        font-weight: 600;
    }}

    /* Chat messages */
    [data-testid="stChatMessage"] {{
        background-color: #FFFFFF !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {BRAND_BORDER};
        border-radius: 14px;
        padding: 0.8rem;
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        color: {TEXT_COLOR} !important;
    }}

    /* Chat input */
    [data-testid="stChatInput"] {{
        background-color: #FFFFFF !important;
        border: 1px solid {BRAND_BORDER};
        border-radius: 14px;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: {BRAND_RED};
        box-shadow: 0 0 0 1px {BRAND_RED};
    }}

    [data-testid="stChatInput"] textarea {{
        background-color: #FFFFFF !important;
        color: {TEXT_COLOR} !important;
        -webkit-text-fill-color: {TEXT_COLOR} !important;
        caret-color: {BRAND_RED} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: #716363 !important;
        -webkit-text-fill-color: #716363 !important;
        opacity: 1;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {BRAND_RED} !important;
        color: #FFFFFF !important;
        border: 1px solid {BRAND_RED} !important;
        border-radius: 10px;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: {BRAND_DARK} !important;
        color: #FFFFFF !important;
        border-color: {BRAND_DARK} !important;
    }}

    /* Footer */
    .footer-note {{
        text-align: center;
        color: #716363 !important;
        font-size: 0.78rem;
        margin-top: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Branded header
# --------------------------------------------------

st.markdown(
    f"""
    <div class="brand-header">
        {logo_html}
        <div class="brand-label">ALMAMLAKA TV</div>
        <h1>AI Knowledge Assistant</h1>
        <p>
            Ask questions about the supplied project documents.
            <br>
            اسأل عن وثائق المشروع باللغة العربية أو الإنجليزية.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    if logo_html:
        st.markdown(
            f"""
            <div style="text-align:center;padding:0.5rem 0 1rem;">
                {logo_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("About this assistant")

    st.write(
        "An AI assistant that answers questions using the "
        "provided project documents and cites its sources."
    )

    st.caption("Powered by retrieval-augmented generation (RAG).")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Almamlaka TV | Project Knowledge Assistant")


# --------------------------------------------------
# Initialize conversation history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Connect to Qdrant
# --------------------------------------------------

if "qdrant_client" not in st.session_state:
    try:
        st.session_state.qdrant_client = get_qdrant_client()
    except Exception:
        st.error(
            "Could not connect to the knowledge database. "
            "Please check the application configuration and try again."
        )
        st.stop()


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Accept a new question
# --------------------------------------------------

question = st.chat_input("Ask about the project... / اسأل عن المشروع...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Almamlaka AI is preparing your answer... / مساعد المملكة يُحضّر إجابتك..."
        ):
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


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
        Answers are based on the supplied documents and should include
        source citations.
        <br>
        الإجابات مستندة إلى الوثائق المقدمة.
    </div>
    """,
    unsafe_allow_html=True,
)
