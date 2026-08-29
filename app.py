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
/* ---------- Font Import & Ultra-Crisp HD Typography ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    color: #F8FAFC !important;
    font-size: 16px;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

/* Premium Dark Slate Background with Softer, Subtler Lighting Tints (Zero Harsh Pink Glare) */
.stApp {
    background: 
        radial-gradient(circle at 10% 12%, rgba(59, 130, 246, 0.18), transparent 45%),
        radial-gradient(circle at 90% 15%, rgba(99, 102, 241, 0.16), transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(14, 165, 233, 0.14), transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(59, 130, 246, 0.10), transparent 45%),
        #0B0F19 !important;
    background-attachment: fixed !important;
}

/* Hide Default Hamburger Menu & Footer, Keep Sidebar Toggle Visible */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="collapsedControl"] { 
    display: flex !important;
    visibility: visible !important;
    color: #38BDF8 !important;
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}
div[data-testid="stDecoration"] { display: none; }

/* ---------- Typography Hierarchy (Pure Crisp White, Maximum Legibility) ---------- */
h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.025em !important;
    line-height: 1.2 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
h2 {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
}
h3 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
h4, h5 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
}
p, span, label, div {
    font-size: 1.02rem;
    line-height: 1.65;
    color: #F8FAFC !important;
}

/* ---------- Buttons & Interactive Elements ---------- */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #06B6D4 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.35rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(37, 99, 235, 0.55), 0 0 15px rgba(6, 182, 212, 0.35) !important;
    transform: translateY(-2px) scale(1.01) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.99) !important;
}

.stButton > button:disabled {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #64748B !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Secondary Button Style (Clean Soft Tint) */
.secondary-btn > button {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    color: #F8FAFC !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
}
.secondary-btn > button:hover {
    background: rgba(59, 130, 246, 0.2) !important;
    border-color: #38BDF8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
}

/* Danger / Delete Button Style */
.danger-btn > button {
    background: rgba(239, 68, 68, 0.16) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    color: #FCA5A5 !important;
    box-shadow: none !important;
    padding: 0.45rem 0.8rem !important;
    font-size: 0.92rem !important;
}
.danger-btn > button:hover {
    background: rgba(239, 68, 68, 0.32) !important;
    border-color: #EF4444 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
}

/* History Item Button */
.history-btn > button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #E2E8F0 !important;
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
    background: rgba(59, 130, 246, 0.16) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #FFFFFF !important;
    transform: translateX(2px) !important;
}

/* Inputs Styling */
.stTextInput input {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-size: 1.02rem !important;
    padding: 0.7rem 0.95rem !important;
    transition: all 0.2s ease !important;
}
.stTextInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.3) !important;
}
.stTextInput label {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}

/* ---------- Sidebar Styling ---------- */
section[data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.95) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* File Uploader Container */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38BDF8 !important;
    background: rgba(56, 189, 248, 0.06) !important;
}
[data-testid="stFileUploader"] small {
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.85rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38BDF8;
    margin: 1.4rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* History Date Subheading */
.history-date-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin: 0.7rem 0 0.35rem 0;
    padding-bottom: 0.2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

/* Document Item Card in Sidebar */
.doc-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}
.doc-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: #38BDF8;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.15);
}
.doc-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #FFFFFF;
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
    gap: 0.5rem;
    margin-top: 0.4rem;
    font-size: 0.84rem;
}
.doc-pill-pages {
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(59, 130, 246, 0.45);
    color: #93C5FD;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 700;
}
.doc-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #E2E8F0;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 600;
}

/* System Stats Box */
.stats-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 0.9rem;
    margin-top: 0.9rem;
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
}
.stat-item {
    text-align: center;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.65rem 0.35rem;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stat-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #38BDF8;
}
.stat-lbl {
    font-size: 0.76rem;
    text-transform: uppercase;
    color: #94A3B8;
    font-weight: 600;
    letter-spacing: 0.06em;
}

/* ---------- LANDING PAGE / AUTH HERO (Split Screen) ---------- */
.landing-container {
    padding: 1.5rem 1rem 3rem 1rem;
    max-width: 1260px;
    margin: 0 auto;
}

.hero-wrapper {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.14);
    border-radius: 28px;
    padding: 3rem 2.6rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #93C5FD;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 1.1rem;
}

.hero-title {
    font-size: 2.85rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    color: #FFFFFF;
}

.hero-subtitle {
    font-size: 1.35rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 0.9rem;
}

.hero-desc {
    font-size: 1.05rem;
    line-height: 1.65;
    color: #CBD5E1;
    margin-bottom: 1.8rem;
}

/* Pipeline Visualizer Bar */
.rag-flow-container {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.6rem;
}
.rag-flow-title {
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38BDF8;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.rag-flow-steps {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
}
.rag-step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 0.45rem 0.7rem;
    font-size: 0.84rem;
    font-weight: 700;
    color: #FFFFFF;
}
.rag-step-arrow {
    color: #38BDF8;
    font-weight: 800;
    font-size: 0.9rem;
}

/* Feature Cards Grid */
.hero-features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}
.hero-feat-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    transition: all 0.2s ease;
}
.hero-feat-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: #38BDF8;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
    transform: translateY(-2px);
}
.hero-feat-icon {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38BDF8;
    margin-bottom: 0.35rem;
}
.hero-feat-name {
    font-weight: 700;
    font-size: 1.02rem;
    color: #FFFFFF;
    margin-bottom: 0.25rem;
}
.hero-feat-desc {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* Login Card (Left Side) */
.auth-card-wrap {
    background: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(24px);
    border: 1.5px solid rgba(255, 255, 255, 0.14);
    border-radius: 26px;
    padding: 2.5rem 2.2rem;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5);
    position: relative;
}
.auth-badge-icon {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.85rem;
    margin: 0 auto 1.3rem auto;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
}
.auth-card-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 0.35rem;
}
.auth-card-subtitle {
    font-size: 0.95rem;
    color: #94A3B8;
    text-align: center;
    margin-bottom: 1.6rem;
}
.auth-alert-error {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #FCA5A5;
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    font-size: 0.92rem;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.auth-security-note {
    font-size: 0.82rem;
    color: #64748B;
    text-align: center;
    margin-top: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}

/* ---------- WORKSPACE TOP HEADER ---------- */
.workspace-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
.workspace-header h1 {
    font-size: 1.8rem !important;
    margin: 0 !important;
    color: #FFFFFF !important;
}
.workspace-header p {
    font-size: 1rem;
    color: #94A3B8;
    margin: 0.3rem 0 0 0;
}

/* Empty State / Welcome Guide */
.empty-workspace-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1.5px dashed rgba(255, 255, 255, 0.14);
    border-radius: 22px;
    padding: 3.2rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 0.9rem;
    display: inline-block;
    color: #38BDF8;
}
.empty-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.45rem;
}
.empty-desc {
    font-size: 1.04rem;
    color: #94A3B8;
    max-width: 550px;
    margin: 0 auto 1.6rem auto;
    line-height: 1.6;
}
.empty-steps {
    display: inline-flex;
    gap: 1.6rem;
    text-align: left;
    background: rgba(255, 255, 255, 0.04);
    padding: 1rem 1.5rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.94rem;
    font-weight: 600;
    color: #E2E8F0;
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
    font-size: 0.84rem;
}

/* ---------- CHAT & MESSAGE BUBBLES (Matching Fig C.2 Layout) ---------- */
/* User Message Bubble (Right-Aligned, Light Blue Pill) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: rgba(37, 99, 235, 0.2) !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 1.1rem 1.4rem !important;
    margin-bottom: 1.3rem !important;
    margin-left: auto !important;
    max-width: 85% !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
}

/* Assistant Message Bubble (Left-Aligned, Dark Slate Card with Blue Accent) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-left: 4px solid #38BDF8 !important;
    border-radius: 18px !important;
    padding: 1.3rem 1.6rem !important;
    margin-bottom: 1.4rem !important;
    margin-right: auto !important;
    max-width: 95% !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35) !important;
}

/* Text Legibility inside Chat Messages (Pure Crisp White, Maximum Sharpness) */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #F8FAFC !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}
div[data-testid="stChatMessage"] strong {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    color: #FFFFFF !important;
    margin-top: 0.9rem !important;
    margin-bottom: 0.45rem !important;
}
div[data-testid="stChatMessage"] code {
    background: rgba(0, 0, 0, 0.4) !important;
    color: #38BDF8 !important;
    padding: 0.2rem 0.45rem !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* ---------- SOURCE CITATIONS ---------- */
.sources-title {
    font-size: 0.85rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38BDF8;
    margin-top: 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #E0F2FE !important;
    border-radius: 8px;
    padding: 0.35rem 0.85rem;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.2rem 0.4rem 0.2rem 0;
    transition: all 0.2s ease;
}
.source-chip:hover {
    background: rgba(56, 189, 248, 0.22);
    border-color: #38BDF8;
    color: #FFFFFF !important;
}
.source-page-badge {
    background: rgba(59, 130, 246, 0.35);
    color: #BFDBFE;
    font-weight: 800;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
    font-size: 0.82rem;
}

/* ---------- ANSWER VERIFICATION CARD ---------- */
.verify-container {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 0.9rem;
}
.verify-summary-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.85rem;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
}
.badge-high-confidence {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #4ADE80;
}
.badge-moderate-confidence {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #FCD34D;
}
.verify-row-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 0.95rem;
}
.verify-row-item:last-child {
    border-bottom: none;
}
.verify-sim-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 0.15rem 0.45rem;
    border-radius: 5px;
    min-width: 46px;
    text-align: center;
}
.verify-status-supported {
    color: #4ADE80;
    font-weight: 700;
}
.verify-status-review {
    color: #FCD34D;
    font-weight: 700;
}

/* Expanders */
details {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 12px !important;
    margin-top: 0.8rem !important;
}
summary {
    color: #38BDF8 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 0.8rem !important;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    background: rgba(11, 15, 25, 0.95) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stChatInput"] textarea {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.16) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    font-size: 1.02rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
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
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #E2E8F0 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: rgba(59, 130, 246, 0.25) !important;
    border-color: #38BDF8 !important;
    color: #FFFFFF !important;
}

/* Footer Tagline */
.app-footer {
    text-align: center;
    color: #64748B;
    font-size: 0.84rem;
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
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
    st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#38BDF8; margin-bottom:0.65rem; letter-spacing:0.06em;'>✦ QUICK PROMPTS</div>", unsafe_allow_html=True)
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

