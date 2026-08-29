import sys
import asyncio

# Fix for Windows asyncio event loop closed error during Streamlit script reruns
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import warnings
import logging

# Suppress harmless deprecation warnings & pypdf repair logs in terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

import streamlit as st
import tempfile
import os
import re
import shutil
import hmac
import time
import bcrypt
from datetime import datetime, date
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from rag_pipeline_v2 import PDFRAGPipelineMistral
from verification_layer import verify_answer
from db import init_db, save_chat_turn, load_user_history, clear_user_history
from export_utils import (
    export_turn_to_txt, export_turn_to_pdf, export_turn_to_docx,
    export_conversation_to_txt, export_conversation_to_pdf, export_conversation_to_docx
)

# Load environment variables
load_dotenv()

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "")

st.set_page_config(
    page_title="Chat with Your PDF · AI Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize SQLite Database
init_db()


# ============================================================
# UTILITY & HELPER FUNCTIONS
# ============================================================
def check_credentials(username: str, password: str) -> bool:
    """Team Passcode Mode: Verifies master access passcode and isolates chat history by entered username."""
    if not username or not username.strip() or not APP_PASSWORD_HASH:
        return False
    try:
        pass_ok = bcrypt.checkpw(password.encode("utf-8"), APP_PASSWORD_HASH.encode("utf-8"))
    except (ValueError, TypeError):
        pass_ok = False
    return pass_ok


def format_file_size(size_bytes: int) -> str:
    """Formats bytes into human-readable string (KB/MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_pdf_page_count(file_path: str) -> int:
    """Calculates total pages in a PDF file using PyPDFLoader."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return len(pages)
    except Exception:
        return 1


def reindex_pipeline(pdf_paths: list):
    """Rebuilds the RAG pipeline with the given list of PDF paths."""
    if not pdf_paths:
        st.session_state.pipeline = None
        return None
    try:
        pipeline = PDFRAGPipelineMistral(pdf_paths=pdf_paths)
        pipeline.setup(force_rebuild=True)
        return pipeline
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        return None
    except Exception as e:
        st.error(f"⚠️ Indexing Error: {str(e)}")
        return None



def format_history_date(ts_str: str) -> str:
    """Groups timestamps into Today, Yesterday, or specific date."""
    try:
        dt = datetime.strptime(ts_str[:10], "%Y-%m-%d").date()
        today = date.today()
        if dt == today:
            return "Today"
        elif (today - dt).days == 1:
            return "Yesterday"
        else:
            return dt.strftime("%b %d, %Y")
    except Exception:
        return "Recent"


# ============================================================
# CUSTOM CSS — RADIANT PALETTE (PINK, BLUE, PURPLE), TYPOGRAPHY & ANIMATIONS
# ============================================================
st.markdown("""
<style>
/* ---------- Font Import & Ultra-Crisp Light HD Styling ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    color: #0F172A !important;
    font-size: 16px;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

/* Clean, Light, Modern App Background (No dark/pink tints) */
.stApp {
    background: #F8FAFC !important;
}

/* Hide Default Streamlit Clutter */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="collapsedControl"] { 
    display: flex !important;
    visibility: visible !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}
div[data-testid="stDecoration"] { display: none; }

/* ---------- Typography Hierarchy (High Contrast, Crisp & Easy to Read) ---------- */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
    line-height: 1.25 !important;
}
h2 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}
h4, h5 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
p, span, label, div {
    font-size: 1rem;
    line-height: 1.6;
    color: #334155 !important;
}

/* ---------- Modern Clean Buttons ---------- */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #1D4ED8 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.98rem !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.18) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

.stButton > button:disabled {
    background: #E2E8F0 !important;
    color: #94A3B8 !important;
    border-color: #CBD5E1 !important;
    box-shadow: none !important;
}

/* Secondary Button Style (Clean White) */
.secondary-btn > button {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    color: #1E293B !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
}
.secondary-btn > button:hover {
    background: #F1F5F9 !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.12) !important;
}

/* Danger / Delete Button Style */
.danger-btn > button {
    background: #FEE2E2 !important;
    border: 1px solid #FCA5A5 !important;
    color: #DC2626 !important;
    box-shadow: none !important;
    padding: 0.45rem 0.8rem !important;
    font-size: 0.9rem !important;
}
.danger-btn > button:hover {
    background: #FEE2E2 !important;
    border-color: #DC2626 !important;
    color: #B91C1C !important;
}

/* History Item Button */
.history-btn > button {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #334155 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.history-btn > button:hover {
    background: #EFF6FF !important;
    border-color: #BFDBFE !important;
    color: #2563EB !important;
}

/* Text Inputs */
.stTextInput input {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    font-size: 1rem !important;
    padding: 0.65rem 0.9rem !important;
}
.stTextInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}
.stTextInput label {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

/* ---------- Sidebar Styling (Crisp White / Light Slate) ---------- */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* File Uploader Container */
[data-testid="stFileUploader"] {
    background: #F8FAFC !important;
    border: 1.5px dashed #94A3B8 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
}
[data-testid="stFileUploader"] small {
    color: #64748B !important;
    font-size: 0.88rem !important;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2563EB;
    margin: 1.4rem 0 0.55rem 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* History Date Subheading */
.history-date-label {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin: 0.65rem 0 0.3rem 0;
    padding-bottom: 0.2rem;
    border-bottom: 1px solid #E2E8F0;
}

/* Document Item Card in Sidebar */
.doc-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.45rem;
    transition: all 0.2s ease;
}
.doc-card:hover {
    background: #EFF6FF;
    border-color: #BFDBFE;
}
.doc-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #0F172A;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.doc-meta-tags {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.35rem;
    font-size: 0.82rem;
}
.doc-pill-pages {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    padding: 0.12rem 0.45rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
}
.doc-pill {
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    color: #475569;
    padding: 0.12rem 0.45rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* System Stats Box */
.stats-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.85rem;
    margin-top: 0.85rem;
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
}
.stat-item {
    text-align: center;
    background: #FFFFFF;
    padding: 0.6rem 0.3rem;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
}
.stat-val {
    font-size: 1.3rem;
    font-weight: 800;
    color: #2563EB;
}
.stat-lbl {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #64748B;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ---------- WORKSPACE TOP HEADER ---------- */
.workspace-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.workspace-header h1 {
    font-size: 1.75rem !important;
    margin: 0 !important;
    color: #0F172A !important;
}
.workspace-header p {
    font-size: 0.98rem;
    color: #64748B;
    margin: 0.25rem 0 0 0;
}

/* Empty State / Welcome Guide */
.empty-workspace-card {
    background: #FFFFFF;
    border: 1.5px dashed #CBD5E1;
    border-radius: 18px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.empty-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
    display: inline-block;
    color: #2563EB;
}
.empty-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.4rem;
}
.empty-desc {
    font-size: 1rem;
    color: #64748B;
    max-width: 540px;
    margin: 0 auto 1.6rem auto;
    line-height: 1.6;
}
.empty-steps {
    display: inline-flex;
    gap: 1.5rem;
    text-align: left;
    background: #F8FAFC;
    padding: 0.9rem 1.4rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: #1E293B;
}
.step-num {
    background: #2563EB;
    color: white;
    font-weight: 800;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
}

/* ---------- CHAT & MESSAGE BUBBLES (Matching Fig C.2) ---------- */
/* User Message Bubble (Right-Aligned, Light Blue) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 1rem 1.35rem !important;
    margin-bottom: 1.2rem !important;
    margin-left: auto !important;
    max-width: 85% !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08) !important;
}

/* Assistant Message Bubble (Left-Aligned, Crisp White with Blue Accent) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #2563EB !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 1.4rem !important;
    margin-right: auto !important;
    max-width: 95% !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
}

/* Text Legibility inside Chat Messages */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #0F172A !important;
    font-size: 1.04rem !important;
    line-height: 1.65 !important;
}
div[data-testid="stChatMessage"] strong {
    color: #0F172A !important;
    font-weight: 700 !important;
}
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    color: #0F172A !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stChatMessage"] code {
    background: #F1F5F9 !important;
    color: #2563EB !important;
    padding: 0.2rem 0.45rem !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.94rem !important;
    border: 1px solid #E2E8F0 !important;
}

/* ---------- SOURCE CITATIONS (Clean Pills) ---------- */
.sources-title {
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2563EB;
    margin-top: 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    color: #1E293B !important;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.88rem;
    font-weight: 600;
    margin: 0.2rem 0.4rem 0.2rem 0;
    transition: all 0.2s ease;
}
.source-chip:hover {
    background: #EFF6FF;
    border-color: #2563EB;
    color: #2563EB !important;
}
.source-page-badge {
    background: #DBEAFE;
    color: #1E40AF;
    font-weight: 800;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
    font-size: 0.82rem;
}

/* ---------- ANSWER VERIFICATION CARD (High Contrast) ---------- */
.verify-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.9rem;
}
.verify-summary-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.8rem;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
}
.badge-high-confidence {
    background: #DCFCE7;
    border: 1px solid #86EFAC;
    color: #166534;
}
.badge-moderate-confidence {
    background: #FEF3C7;
    border: 1px solid #FCD34D;
    color: #92400E;
}
.verify-row-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #E2E8F0;
    font-size: 0.94rem;
}
.verify-row-item:last-child {
    border-bottom: none;
}
.verify-sim-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: #2563EB;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 0.15rem 0.45rem;
    border-radius: 5px;
    min-width: 46px;
    text-align: center;
}
.verify-status-supported {
    color: #15803D;
    font-weight: 700;
}
.verify-status-review {
    color: #B45309;
    font-weight: 700;
}

/* Expanders */
details {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    margin-top: 0.8rem !important;
}
summary {
    color: #1E293B !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    padding: 0.5rem 0.8rem !important;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border-top: 1px solid #E2E8F0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #F8FAFC !important;
    border: 1.5px solid #CBD5E1 !important;
    color: #0F172A !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    background: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #94A3B8 !important;
    font-size: 0.98rem !important;
}

/* Download Buttons inside Messages */
.message-actions-bar {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 0.9rem;
    padding-top: 0.6rem;
    border-top: 1px solid #F1F5F9;
}
.stDownloadButton > button {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    color: #334155 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}

/* Footer Tagline */
.app-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.85rem;
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #E2E8F0;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "entered" not in st.session_state:
    st.session_state.entered = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "auth_error" not in st.session_state:
    st.session_state.auth_error = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "docs_metadata" not in st.session_state:
    st.session_state.docs_metadata = {}  # {path: {"name": str, "pages": int, "size": int}}
if "temp_storage_dir" not in st.session_state:
    st.session_state.temp_storage_dir = tempfile.mkdtemp(prefix="chat_pdf_")

SUGGESTED_QUESTIONS = [
    "✦ Summarize this document",
    "✦ What are the key points?",
    "✦ List the main topics covered",
    "✦ Identify actionable takeaways",
]


# ============================================================
# LANDING PAGE & AUTHENTICATION (SPLIT SCREEN 2-COLUMN)
# ============================================================
if not st.session_state.entered:
    st.markdown("<div class='landing-container'>", unsafe_allow_html=True)
    
    col_login, col_hero = st.columns([1, 1.45], gap="large")

    # LEFT COLUMN: Compact Login Card
    with col_login:
        st.markdown(
            '<div class="auth-card-wrap">'
            '<div class="auth-badge-icon">◫</div>'
            '<div class="auth-card-title">Welcome Back</div>'
            '<div class="auth-card-subtitle">Enter your name &amp; team passcode to access your workspace</div>',
            unsafe_allow_html=True,
        )

        if not APP_PASSWORD_HASH:
            st.markdown(
                '<div class="auth-alert-error">'
                '✕ <span>APP_PASSWORD_HASH missing in .env configuration.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.auth_error:
            st.markdown(
                '<div class="auth-alert-error">'
                '✕ <span>Invalid passcode or empty username. Please check your credentials.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Your Name / Username", key="login_username", placeholder="e.g. John, Sarah, Admin")
            password_input = st.text_input("Access Passcode", type="password", key="login_password", placeholder="Enter team passcode")
            submitted = st.form_submit_button("➔ Enter Workspace", use_container_width=True)

        if submitted:
            if check_credentials(username_input, password_input):
                st.session_state.entered = True
                st.session_state.username = username_input.strip()
                st.session_state.auth_error = False
                # Load persistent SQLite history for this user
                st.session_state.history = load_user_history(st.session_state.username)
                st.rerun()
            else:
                st.session_state.auth_error = True
                st.rerun()

        st.markdown(
            '<div class="auth-security-note">'
            '🛡 Protected Workspace · Constant-Time Auth & SQLite Persistence'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # RIGHT COLUMN: Hero Presentation Showcase with RAG Animation & Floating Badges
    with col_hero:
        hero_html = (
            '<div class="hero-wrapper">'
            '<div class="floating-badge-1">◫ PDF Vector Search</div>'
            '<div class="floating-badge-2">🛡 Grounded AI</div>'
            '<div class="hero-tag">✦ Next-Gen Document Intelligence</div>'
            '<div class="hero-title">Chat With Your PDF</div>'
            '<div class="hero-subtitle">Multi-Document Question & Answering with Source Citation</div>'
            '<div class="hero-desc">'
            'Transform complex PDF reports, research papers, legal documents, and corporate disclosures into an interactive conversational knowledge base. '
            'Get precise, verified answers backed by exact page citations.'
            '</div>'
            '<div class="rag-flow-container">'
            '<div class="rag-flow-title">◈ Neural RAG Intelligence Pipeline</div>'
            '<div class="rag-flow-steps">'
            '<div class="rag-step-node">◫ PDF Input</div>'
            '<div class="rag-step-arrow">➔</div>'
            '<div class="rag-step-node">✂ Chunks</div>'
            '<div class="rag-step-arrow">➔</div>'
            '<div class="rag-step-node">◈ Embeddings</div>'
            '<div class="rag-step-arrow">➔</div>'
            '<div class="rag-step-node">⚡ FAISS Index</div>'
            '<div class="rag-step-arrow">➔</div>'
            '<div class="rag-step-node">✦ Verified Answer</div>'
            '</div>'
            '</div>'
            '<div class="hero-features-grid">'
            '<div class="hero-feat-card">'
            '<div class="hero-feat-icon">◫</div>'
            '<div class="hero-feat-name">Multi-PDF Processing</div>'
            '<div class="hero-feat-desc">Simultaneously index multiple documents with FAISS vector similarity search.</div>'
            '</div>'
            '<div class="hero-feat-card">'
            '<div class="hero-feat-icon">⎘</div>'
            '<div class="hero-feat-name">Exact Page Citations</div>'
            '<div class="hero-feat-desc">Every response links directly to source document titles and precise page numbers.</div>'
            '</div>'
            '<div class="hero-feat-card">'
            '<div class="hero-feat-icon">🛡</div>'
            '<div class="hero-feat-name">Grounding Verification</div>'
            '<div class="hero-feat-desc">Automated sentence-by-sentence cosine verification layer to prevent hallucinations.</div>'
            '</div>'
            '<div class="hero-feat-card">'
            '<div class="hero-feat-icon">↓</div>'
            '<div class="hero-feat-name">Summary & Export</div>'
            '<div class="hero-feat-desc">Generate instant executive summaries and export full verified reports in one click.</div>'
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# MAIN SIDEBAR — DOCUMENT MANAGEMENT, CHAT HISTORY & ACTIONS
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.65rem; margin-bottom: 0.2rem;">'
        '<div style="font-size:1.8rem; color:#00D2FF;">◫</div>'
        '<div>'
        '<div style="font-weight:800; font-size:1.25rem; color:#FFFFFF; line-height:1.2;">Chat with Your PDF</div>'
        '<div style="font-size:0.82rem; color:#00D2FF; font-weight:600;">AI Document Intelligence</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    st.divider()

    # SECTION 1: UPLOAD DOCUMENTS
    st.markdown("<div class='sidebar-section-title'>⇪ Upload Documents</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="pdf_uploader"
    )

    process_btn = st.button(
        f"✦ Process & Index {f'({len(uploaded_files)})' if uploaded_files else ''}",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_btn and uploaded_files:
        anim_placeholder = st.empty()
        
        anim_steps = [
            ("Reading Documents...", 20),
            ("Creating Text Chunks...", 45),
            ("Generating Mistral Embeddings...", 70),
            ("Building FAISS Vector Index...", 90),
            ("Ready! Indexing Complete.", 100),
        ]
        
        new_paths = []
        for i, f in enumerate(uploaded_files):
            target_path = os.path.join(st.session_state.temp_storage_dir, f.name)
            with open(target_path, "wb") as out:
                out.write(f.getbuffer())
            
            page_count = get_pdf_page_count(target_path)
            file_size = len(f.getbuffer())
            
            st.session_state.docs_metadata[target_path] = {
                "name": f.name,
                "pages": page_count,
                "size": file_size,
            }
            new_paths.append(target_path)

        for step_text, pct in anim_steps:
            anim_placeholder.progress(pct, text=f"◈ {step_text}")
            time.sleep(0.2)
            
        all_active_paths = list(st.session_state.docs_metadata.keys())
        new_pipeline = reindex_pipeline(all_active_paths)
        
        time.sleep(0.2)
        anim_placeholder.empty()
        
        if new_pipeline is not None:
            st.session_state.pipeline = new_pipeline
            st.toast(f"Successfully indexed {len(all_active_paths)} document(s)!", icon="✅")
            st.rerun()
        else:
            st.toast("Indexing failed: No valid text found in uploaded PDF(s).", icon="⚠️")


    # SECTION 2: LOADED DOCUMENTS MANAGEMENT
    if st.session_state.docs_metadata:
        st.markdown("<div class='sidebar-section-title'>◫ Loaded Documents</div>", unsafe_allow_html=True)
        
        paths_to_delete = []
        for doc_path, meta in list(st.session_state.docs_metadata.items()):
            col_info, col_del = st.columns([4, 1.2])
            with col_info:
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-title" title="{meta["name"]}">◫ {meta["name"]}</div>'
                    f'<div class="doc-meta-tags">'
                    f'<span class="doc-pill-pages">{meta["pages"]} {"page" if meta["pages"] == 1 else "pages"}</span>'
                    f'<span class="doc-pill">{format_file_size(meta["size"])}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                st.markdown("<div class='danger-btn' style='margin-top: 0.35rem;'>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_{doc_path}", help=f"Remove {meta['name']}"):
                    paths_to_delete.append(doc_path)
                st.markdown("</div>", unsafe_allow_html=True)

        if paths_to_delete:
            for p in paths_to_delete:
                if p in st.session_state.docs_metadata:
                    del st.session_state.docs_metadata[p]
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            
            remaining_paths = list(st.session_state.docs_metadata.keys())
            if remaining_paths:
                with st.spinner("Re-indexing remaining documents..."):
                    st.session_state.pipeline = reindex_pipeline(remaining_paths)
                st.toast(f"Updated index with {len(remaining_paths)} document(s).", icon="✅")
            else:
                st.session_state.pipeline = None
                st.toast("All documents removed. Index cleared.", icon="🗑️")
            st.rerun()

        # CLEAR ALL DOCUMENTS
        st.markdown("<div class='danger-btn' style='margin-top: 0.6rem;'>", unsafe_allow_html=True)
        if st.button("⟳ Clear All Documents", use_container_width=True):
            st.session_state.docs_metadata = {}
            st.session_state.pipeline = None
            try:
                shutil.rmtree(st.session_state.temp_storage_dir, ignore_errors=True)
                st.session_state.temp_storage_dir = tempfile.mkdtemp(prefix="chat_pdf_")
            except Exception:
                pass
            st.toast("Workspace reset: all documents and vector index cleared.", icon="🧹")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # SECTION 3: SYSTEM METRICS & STATS
        total_docs = len(st.session_state.docs_metadata)
        total_pages = sum(m["pages"] for m in st.session_state.docs_metadata.values())
        
        st.markdown(
            f'<div class="stats-box">'
            f'<div class="stats-grid">'
            f'<div class="stat-item">'
            f'<div class="stat-val">{total_docs}</div>'
            f'<div class="stat-lbl">Documents</div>'
            f'</div>'
            f'<div class="stat-item">'
            f'<div class="stat-val">{total_pages}</div>'
            f'<div class="stat-lbl">Total Pages</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # SECTION 4: PERSISTENT CHAT HISTORY (SQLITE)
    st.markdown("<div class='sidebar-section-title'>⏱ Chat History</div>", unsafe_allow_html=True)
    
    if st.session_state.history:
        # Group by date categories
        history_by_date = {}
        for turn in reversed(st.session_state.history):
            d_group = format_history_date(turn.get("timestamp", ""))
            if d_group not in history_by_date:
                history_by_date[d_group] = []
            history_by_date[d_group].append(turn)
            
        for date_lbl, turns in history_by_date.items():
            st.markdown(f"<div class='history-date-label'>{date_lbl}</div>", unsafe_allow_html=True)
            for t in turns[:8]:  # show up to 8 recent per group
                q_text = t["question"]
                short_q = q_text if len(q_text) <= 28 else q_text[:28] + "..."
                st.markdown("<div class='history-btn' style='margin-bottom:0.25rem;'>", unsafe_allow_html=True)
                if st.button(f"💬 {short_q}", key=f"hist_btn_{t.get('id', t['timestamp'])}", help=q_text, use_container_width=True):
                    # Set pending question or view
                    st.session_state.pending_question = q_text
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='secondary-btn' style='margin-top:0.4rem; margin-bottom:0.6rem;'>", unsafe_allow_html=True)
        if st.button("⟳ Clear History", use_container_width=True):
            clear_user_history(st.session_state.username)
            st.session_state.history = []
            st.toast("Chat history cleared from database.", icon="🧹")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:0.80rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; color:#38BDF8; margin-bottom:0.35rem;'>↓ EXPORT ALL CHATS</div>", unsafe_allow_html=True)
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            pdf_conv = export_conversation_to_pdf(st.session_state.username, st.session_state.history)
            st.download_button(
                label="PDF",
                data=pdf_conv,
                file_name=f"chat_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_e2:
            docx_conv = export_conversation_to_docx(st.session_state.username, st.session_state.history)
            st.download_button(
                label="DOCS",
                data=docx_conv,
                file_name=f"chat_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        with col_e3:
            txt_conv = export_conversation_to_txt(st.session_state.username, st.session_state.history)
            st.download_button(
                label="Text",
                data=txt_conv,
                file_name=f"chat_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.markdown("<p style='font-size:0.88rem; color:#8C84A0; margin:0.3rem 0;'>No previous conversation recorded.</p>", unsafe_allow_html=True)

    # SECTION 5: ACCOUNT & LOGOUT
    st.markdown("<div class='sidebar-section-title'>👤 Account</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.92rem; color:#DDD8EB; margin-bottom:0.5rem;">Signed in as <b>{st.session_state.username}</b></div>',
        unsafe_allow_html=True
    )
    st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.entered = False
        st.session_state.username = ""
        st.session_state.auth_error = False
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN WORKSPACE AREA
# ============================================================
# Header Banner
st.markdown(
    '<div class="workspace-header">'
    '<div>'
    '<h1>◫ Chat with Your PDF</h1>'
    '<p>Ask questions across multiple documents with grounding verification and exact page citations.</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# CASE 1: No documents loaded yet
if not st.session_state.pipeline:
    st.markdown(
        '<div class="empty-workspace-card">'
        '<div class="empty-icon">◫</div>'
        '<div class="empty-title">No Documents Loaded</div>'
        '<div class="empty-desc">'
        'Upload one or more PDF files using the left sidebar and click <b>Process & Index</b> to start asking questions.'
        '</div>'
        '<div class="empty-steps">'
        '<div class="step-item"><span class="step-num">1</span> Upload PDF Files</div>'
        '<div class="step-item"><span class="step-num">2</span> Click Process & Index</div>'
        '<div class="step-item"><span class="step-num">3</span> Ask Anything with Citations</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# CASE 2: Documents are loaded — Q&A Interface
else:
    # Quick Suggested Prompt Pills
    st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#FF7EB6; margin-bottom:0.65rem; letter-spacing:0.06em;'>✦ QUICK PROMPTS</div>", unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, sq in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
            if st.button(sq, key=f"sq_{sq}", use_container_width=True):
                st.session_state.pending_question = sq.replace("✦ ", "")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Chat Input Handling
    user_input = st.chat_input("Ask any question about the uploaded documents...")

    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        with st.spinner("Analyzing indexed documents with Mistral & verifying answer..."):
            answer, sources = st.session_state.pipeline.ask(user_input)
            verification = verify_answer(
                answer=answer,
                source_documents=sources,
                embedding_model=st.session_state.pipeline.embedding_model,
            )

            # Filter sources to only include chunks that actually contain verified supporting quotes
            supported_quotes = [
                " ".join(re.sub(r'[^\w\s]', '', r.get("quote", "").lower()).split())
                for r in verification
                if r.get("supported", False) and r.get("quote")
            ]
            if supported_quotes and sources:
                verified_sources = []
                seen_keys = set()
                for doc in sources:
                    doc_text = getattr(doc, "page_content", str(doc)).lower()
                    doc_norm = " ".join(re.sub(r'[^\w\s]', '', doc_text).split())
                    if any(q in doc_norm or q[:25] in doc_norm for q in supported_quotes):
                        p_val = doc.metadata.get("page") if hasattr(doc, "metadata") else doc.get("page")
                        f_val = doc.metadata.get("source_file") if hasattr(doc, "metadata") else doc.get("source_file")
                        k = (f_val, p_val)
                        if k not in seen_keys:
                            verified_sources.append(doc)
                            seen_keys.add(k)
                if verified_sources:
                    sources = verified_sources
            
            # Persist to SQLite Database
            turn_id = save_chat_turn(
                username=st.session_state.username,
                question=user_input,
                answer=answer,
                sources=sources,
                verification=verification
            )
        
        # Append to active session history
        st.session_state.history.append({
            "id": turn_id,
            "question": user_input,
            "answer": answer,
            "sources": sources,
            "verification": verification,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.rerun()

    # Render Chat History (Chronological Top-to-Bottom order, matching ChatGPT)
    for i, entry in enumerate(st.session_state.history):
        # User message
        with st.chat_message("user"):
            st.markdown(f"**{entry['question']}**")

        # Assistant response
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

            # Render Source Chips (Directly populated from unique verified quotes in the Grounding Report)
            raw_sources = entry.get("sources", [])
            verification = entry.get("verification", [])
            
            badges = []
            seen_badge_keys = set()
            for r in verification:
                if r.get("supported", False) and r.get("cited_page"):
                    fname = r.get("source_file") or "Document"
                    page = str(r.get("cited_page"))
                    key = (fname, page)
                    if key not in seen_badge_keys:
                        badges.append({"source_file": fname, "page": page})
                        seen_badge_keys.add(key)

            # Fallback if verification produced no supported badges
            if not badges and raw_sources:
                for doc in raw_sources:
                    fname = doc.get("source_file", "Document") if isinstance(doc, dict) else doc.metadata.get("source_file", "Document")
                    page = str(doc.get("page", "?") if isinstance(doc, dict) else doc.metadata.get("page", "?"))
                    key = (fname, page)
                    if key not in seen_badge_keys:
                        badges.append({"source_file": fname, "page": page})
                        seen_badge_keys.add(key)

            if badges:
                chips_html = ""
                source_lines_plain = []
                for b in badges:
                    fname = b["source_file"]
                    page = b["page"]
                    chips_html += f"<span class='source-chip'>⎘ {fname} <span class='source-page-badge'>p. {page}</span></span>"
                    source_lines_plain.append(f"- {fname}, page {page}")

                st.markdown("<div class='sources-title'>⎘ RETRIEVED SOURCES</div>", unsafe_allow_html=True)
                st.markdown(chips_html, unsafe_allow_html=True)

            # Render Verification Layer
            verification = entry.get("verification", [])
            verification_lines_plain = []
            if verification:
                with st.expander("🛡 Grounding & Verification Report", expanded=False):
                    supported_count = sum(1 for r in verification if r.get("supported", False))
                    total_sentences = len(verification)
                    percent = int((supported_count / total_sentences) * 100) if total_sentences > 0 else 0
                    
                    badge_class = "badge-high-confidence" if percent >= 75 else "badge-moderate-confidence"
                    
                    verify_header_html = (
                        f'<div class="verify-container">'
                        f'<div class="verify-summary-badge {badge_class}">'
                        f'🛡 {percent}% Grounded ({supported_count}/{total_sentences} sentences verified)'
                        f'</div>'
                    )
                    
                    rows_html = ""
                    for r in verification:
                        is_sup = r.get("supported", False)
                        sim = r.get("similarity", 0.0)
                        sent = r.get("sentence", "")
                        quote = r.get("quote", "")
                        status_label = "<span class='verify-status-supported'>✓ Supported</span>" if is_sup else "<span class='verify-status-review'>⚠ Unsupported</span>"
                        status_plain = "Supported" if is_sup else "Unsupported"
                        
                        quote_html = f'<div style="font-size:0.86rem; color:#86EFAC; margin-top:0.25rem; font-style:italic;">💬 Source Quote: "{quote}"</div>' if (is_sup and quote) else '<div style="font-size:0.84rem; color:#FCA5A5; margin-top:0.25rem;">⚠ No direct supporting statement found in retrieved sources.</div>'
                        
                        rows_html += (
                            f'<div class="verify-row-item">'
                            f'<span class="verify-sim-score">{sim:.2f}</span>'
                            f'<div>{status_label} — <span>{sent}</span>{quote_html}</div>'
                            f'</div>'
                        )
                        quote_plain = f' (Quote: "{quote}")' if (is_sup and quote) else ' (No supporting quote in context)'
                        verification_lines_plain.append(f"- [{status_plain}] ({sim:.2f}) {sent}{quote_plain}")
                        
                    st.markdown(verify_header_html + rows_html + "</div>", unsafe_allow_html=True)

            # Turn Download Action: 3 Formats (PDF, DOCS, Text)
            st.markdown("<div class='message-actions-bar'>", unsafe_allow_html=True)
            col_d_lbl, col_d_pdf, col_d_docx, col_d_txt = st.columns([1.5, 1, 1.2, 0.9])
            
            with col_d_lbl:
                st.markdown("<div style='font-size:0.84rem; color:#A5F3FC; font-weight:700; padding-top:0.35rem;'>↓ DOWNLOAD REPORT:</div>", unsafe_allow_html=True)
                
            with col_d_pdf:
                turn_pdf_bytes = export_turn_to_pdf(
                    entry['question'], entry['answer'], entry.get('sources', []), entry.get('verification', []), entry.get('timestamp', '')
                )
                st.download_button(
                    label="↓ PDF",
                    data=turn_pdf_bytes,
                    file_name=f"qa_report_{i+1}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{i}_{len(entry['question'])}_{entry.get('id', '')}",
                    use_container_width=True
                )
                
            with col_d_docx:
                turn_docx_bytes = export_turn_to_docx(
                    entry['question'], entry['answer'], entry.get('sources', []), entry.get('verification', []), entry.get('timestamp', '')
                )
                st.download_button(
                    label="↓ DOCS",
                    data=turn_docx_bytes,
                    file_name=f"qa_report_{i+1}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_docx_{i}_{len(entry['question'])}_{entry.get('id', '')}",
                    use_container_width=True
                )
                
            with col_d_txt:
                turn_txt_str = export_turn_to_txt(
                    entry['question'], entry['answer'], entry.get('sources', []), entry.get('verification', []), entry.get('timestamp', '')
                )
                st.download_button(
                    label="↓ Text",
                    data=turn_txt_str,
                    file_name=f"qa_report_{i+1}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{i}_{len(entry['question'])}_{entry.get('id', '')}",
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="app-footer">'
    '<b>Chat with Your PDF</b> · Multi-Document Question & Answering with Source Citation<br>'
    'Powered by Mistral AI · FAISS Vector Search · SQLite Persistence · LangChain · Streamlit'
    '</div>',
    unsafe_allow_html=True,
)

