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
/* ---------- Font Import & Ultra-Crisp HD Styling ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    color: #FFFFFF !important;
    font-size: 16.5px;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

/* Background with Radiant Tri-Color Harmony (Pink, Blue, Purple) */
.stApp {
    background: 
        radial-gradient(circle at 8% 10%, rgba(255, 46, 147, 0.45), transparent 42%),
        radial-gradient(circle at 92% 14%, rgba(0, 210, 255, 0.40), transparent 44%),
        radial-gradient(circle at 50% 92%, rgba(139, 92, 246, 0.48), transparent 52%),
        radial-gradient(circle at 78% 68%, rgba(236, 72, 153, 0.30), transparent 46%),
        radial-gradient(circle at 22% 78%, rgba(59, 130, 246, 0.32), transparent 48%),
        #07030F;
    background-attachment: fixed;
}

/* Hide Default Hamburger Menu & Footer, but keep Sidebar Toggle always visible */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="collapsedControl"] { 
    display: flex !important;
    visibility: visible !important;
    color: #00D2FF !important;
    background: rgba(255, 46, 147, 0.25) !important;
    border: 1px solid #00D2FF !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}
div[data-testid="stDecoration"] { display: none; }

/* ---------- Typography Hierarchy (Crisp, High-Contrast & Legible) ---------- */
h1 {
    font-size: 2.5rem !important; /* 40px */
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
h2 {
    font-size: 1.65rem !important; /* 26px */
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
}
h3 {
    font-size: 1.35rem !important; /* 22px */
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
h4, h5 {
    font-size: 1.15rem !important; /* 18px */
    font-weight: 600 !important;
    color: #F1F5F9 !important;
}
p, span, label, div {
    font-size: 1.05rem; /* ~17px */
    line-height: 1.65;
    color: #F8FAFC !important;
}

/* ---------- Keyframe Animations (Subtle & Professional) ---------- */
@keyframes floatDoc1 {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-7px) rotate(1.2deg); }
}

@keyframes floatDoc2 {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(7px) rotate(-1.2deg); }
}

@keyframes neuralPulse {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(255, 46, 147, 0.4)); transform: scale(1); }
    50% { filter: drop-shadow(0 0 24px rgba(0, 210, 255, 0.6)); transform: scale(1.02); }
}

@keyframes gradientShimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes fadeInSlideUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ---------- Radiant Buttons & Inputs ---------- */
.stButton > button {
    background: linear-gradient(135deg, #FF2E93 0%, #A855F7 50%, #00D2FF 100%) !important;
    background-size: 200% 200% !important;
    animation: gradientShimmer 6s ease infinite !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.35rem !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 6px 22px rgba(255, 46, 147, 0.35), 0 2px 10px rgba(0, 210, 255, 0.2) !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(255, 46, 147, 0.55), 0 0 20px rgba(0, 210, 255, 0.4) !important;
    transform: translateY(-2px) scale(1.01) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.99) !important;
}

.stButton > button:disabled {
    background: rgba(45, 35, 65, 0.6) !important;
    color: #8C84A0 !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Secondary Button Style */
.secondary-btn > button {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    color: #F1EDFC !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
}
.secondary-btn > button:hover {
    background: rgba(255, 255, 255, 0.14) !important;
    border-color: #00D2FF !important;
    box-shadow: 0 6px 20px rgba(0, 210, 255, 0.3) !important;
    color: #FFFFFF !important;
}

/* Danger / Delete Button Style */
.danger-btn > button {
    background: rgba(239, 68, 68, 0.18) !important;
    border: 1px solid rgba(239, 68, 68, 0.45) !important;
    color: #FCA5A5 !important;
    box-shadow: none !important;
    padding: 0.45rem 0.8rem !important;
    font-size: 0.92rem !important;
}
.danger-btn > button:hover {
    background: rgba(239, 68, 68, 0.35) !important;
    border-color: #EF4444 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.4) !important;
}

/* History Item Button */
.history-btn > button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    color: #DDD8EB !important;
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
    background: rgba(0, 210, 255, 0.12) !important;
    border-color: rgba(0, 210, 255, 0.4) !important;
    color: #FFFFFF !important;
    transform: translateX(2px) !important;
}

/* Inputs Styling */
.stTextInput input {
    background: rgba(20, 14, 32, 0.88) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    padding: 0.7rem 0.95rem !important;
    transition: all 0.2s ease !important;
}
.stTextInput input:focus {
    border-color: #FF2E93 !important;
    box-shadow: 0 0 16px rgba(255, 46, 147, 0.35) !important;
}
.stTextInput label {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #E2DCF0 !important;
}

/* ---------- Sidebar Styling ---------- */
section[data-testid="stSidebar"] {
    background: rgba(12, 7, 22, 0.9) !important;
    backdrop-filter: blur(26px) !important;
    border-right: 1px solid rgba(255, 46, 147, 0.2) !important;
}

/* File Uploader Container */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1.5px dashed rgba(0, 210, 255, 0.45) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #FF2E93 !important;
    background: rgba(255, 46, 147, 0.04) !important;
}
[data-testid="stFileUploader"] small {
    color: #B5ADC9 !important;
    font-size: 0.9rem !important;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.88rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #FF7EB6;
    margin: 1.5rem 0 0.65rem 0;
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
    color: #38BDF8;
    margin: 0.7rem 0 0.35rem 0;
    padding-bottom: 0.2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

/* Document Item Card in Sidebar */
.doc-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}
.doc-card:hover {
    background: rgba(255, 255, 255, 0.09);
    border-color: #00D2FF;
    box-shadow: 0 4px 18px rgba(0, 210, 255, 0.15);
}
.doc-title {
    font-size: 0.98rem;
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
    gap: 0.55rem;
    margin-top: 0.4rem;
    font-size: 0.84rem;
}
.doc-pill-pages {
    background: rgba(255, 46, 147, 0.22);
    border: 1px solid rgba(255, 46, 147, 0.45);
    color: #FFB3D9;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 700;
}
.doc-pill {
    background: rgba(0, 210, 255, 0.18);
    border: 1px solid rgba(0, 210, 255, 0.35);
    color: #A5F3FC;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 600;
}

/* System Stats Box */
.stats-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
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
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF7EB6, #C084FC, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-lbl {
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #B5ADC9;
    font-weight: 600;
    letter-spacing: 0.06em;
}

/* ---------- LANDING PAGE / AUTH HERO (Split Screen) ---------- */
.landing-container {
    padding: 1.5rem 1rem 3rem 1rem;
    max-width: 1260px;
    margin: 0 auto;
}

/* Floating Doc Badges in Hero */
.floating-badge-1 {
    position: absolute;
    top: 25px;
    right: 35px;
    background: rgba(255, 46, 147, 0.18);
    border: 1px solid rgba(255, 46, 147, 0.4);
    color: #FFA6D5;
    padding: 0.45rem 1rem;
    border-radius: 12px;
    font-size: 0.86rem;
    font-weight: 700;
    animation: floatDoc1 5s ease-in-out infinite;
    backdrop-filter: blur(14px);
    box-shadow: 0 6px 20px rgba(255, 46, 147, 0.25);
}

.floating-badge-2 {
    position: absolute;
    bottom: 25px;
    right: 45px;
    background: rgba(0, 210, 255, 0.16);
    border: 1px solid rgba(0, 210, 255, 0.4);
    color: #7DD3FC;
    padding: 0.45rem 1rem;
    border-radius: 12px;
    font-size: 0.86rem;
    font-weight: 700;
    animation: floatDoc2 6s ease-in-out infinite;
    backdrop-filter: blur(14px);
    box-shadow: 0 6px 20px rgba(0, 210, 255, 0.2);
}

.hero-wrapper {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.07) 0%, rgba(255, 46, 147, 0.12) 40%, rgba(0, 210, 255, 0.10) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.16);
    border-radius: 28px;
    padding: 3rem 2.6rem;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.55);
    position: relative;
    overflow: hidden;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, rgba(255, 46, 147, 0.25), rgba(0, 210, 255, 0.25));
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #FFFFFF;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(255, 46, 147, 0.25);
}

.hero-title {
    font-size: 2.85rem; /* ~44px */
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    background: linear-gradient(120deg, #FFFFFF 10%, #FF7EB6 45%, #C084FC 75%, #38BDF8 100%);
    background-size: 200% 200%;
    animation: gradientShimmer 8s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.4rem; /* 22px */
    font-weight: 700;
    color: #F1EDFC;
    margin-bottom: 0.9rem;
}

.hero-desc {
    font-size: 1.08rem; /* 17px */
    line-height: 1.65;
    color: #C5BFD8;
    margin-bottom: 1.8rem;
}

/* Animated RAG Pipeline Visualizer Bar */
.rag-flow-container {
    background: rgba(14, 9, 26, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.8rem;
}
.rag-flow-title {
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #00D2FF;
    margin-bottom: 0.7rem;
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
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    font-size: 0.86rem;
    font-weight: 700;
    color: #FFFFFF;
    transition: all 0.2s ease;
}
.rag-step-node:hover {
    border-color: #FF2E93;
    transform: scale(1.04);
}
.rag-step-arrow {
    color: #FF7EB6;
    font-weight: 800;
    font-size: 0.95rem;
}

/* Hero Feature Cards Grid */
.hero-features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.1rem;
}
.hero-feat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    transition: all 0.25s ease;
}
.hero-feat-card:hover {
    background: rgba(255, 255, 255, 0.09);
    border-color: #FF2E93;
    box-shadow: 0 8px 24px rgba(255, 46, 147, 0.2);
    transform: translateY(-2px);
}
.hero-feat-icon {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00D2FF;
    margin-bottom: 0.4rem;
}
.hero-feat-name {
    font-weight: 700;
    font-size: 1.05rem;
    color: #FFFFFF;
    margin-bottom: 0.3rem;
}
.hero-feat-desc {
    font-size: 0.92rem;
    color: #B8B0CC;
    line-height: 1.5;
}

/* Compact Login Card (Left Side) */
.auth-card-wrap {
    background: rgba(18, 12, 28, 0.88);
    backdrop-filter: blur(28px);
    border: 1.5px solid rgba(255, 255, 255, 0.16);
    border-radius: 28px;
    padding: 2.6rem 2.3rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
    position: relative;
}
.auth-badge-icon {
    width: 68px;
    height: 68px;
    border-radius: 22px;
    background: linear-gradient(135deg, #FF2E93 0%, #A855F7 50%, #00D2FF 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 1.4rem auto;
    animation: neuralPulse 5s ease-in-out infinite;
}
.auth-card-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 0.35rem;
}
.auth-card-subtitle {
    font-size: 0.98rem;
    color: #B8B0CC;
    text-align: center;
    margin-bottom: 1.8rem;
}
.auth-alert-error {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #FCA5A5;
    padding: 0.8rem 0.95rem;
    border-radius: 12px;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.auth-security-note {
    font-size: 0.84rem;
    color: #8C84A0;
    text-align: center;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}

/* ---------- WORKSPACE TOP HEADER ---------- */
.workspace-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 46, 147, 0.10) 50%, rgba(0, 210, 255, 0.08) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    padding: 1.6rem 2.2rem;
    margin-bottom: 1.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3);
}
.workspace-header h1 {
    font-size: 1.85rem !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #FFFFFF 20%, #FF7EB6 60%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.workspace-header p {
    font-size: 1.02rem;
    color: #C5BFD8;
    margin: 0.3rem 0 0 0;
}

/* Empty State / Welcome Guide */
.empty-workspace-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1.5px dashed rgba(255, 255, 255, 0.15);
    border-radius: 24px;
    padding: 3.5rem 2.2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.empty-icon {
    font-size: 3.2rem;
    margin-bottom: 1rem;
    display: inline-block;
    color: #00D2FF;
}
.empty-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.empty-desc {
    font-size: 1.08rem;
    color: #C5BFD8;
    max-width: 560px;
    margin: 0 auto 1.8rem auto;
    line-height: 1.65;
}
.empty-steps {
    display: inline-flex;
    gap: 1.8rem;
    text-align: left;
    background: rgba(255, 255, 255, 0.04);
    padding: 1.1rem 1.6rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.96rem;
    font-weight: 600;
    color: #F1EDFC;
}
.step-num {
    background: linear-gradient(135deg, #FF2E93, #00D2FF);
    color: white;
    font-weight: 800;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.86rem;
}

/* ---------- CHAT & MESSAGE BUBBLES ---------- */
/* Assistant Message Card */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(20, 13, 33, 0.88) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-left: 4px solid #FF2E93 !important;
    border-radius: 20px !important;
    padding: 1.35rem 1.6rem !important;
    margin-bottom: 1.4rem !important;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4) !important;
    animation: fadeInSlideUp 0.35s ease-out !important;
}

/* User Message Card */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: rgba(45, 20, 68, 0.72) !important;
    border: 1px solid rgba(255, 46, 147, 0.35) !important;
    border-radius: 20px !important;
    padding: 1.15rem 1.45rem !important;
    margin-bottom: 1.4rem !important;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3) !important;
    animation: fadeInSlideUp 0.3s ease-out !important;
}

/* Text Legibility inside Chat Messages */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #F8F6FD !important;
    font-size: 1.08rem !important;
    line-height: 1.72 !important;
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
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}
div[data-testid="stChatMessage"] code {
    background: rgba(0, 0, 0, 0.45) !important;
    color: #FF7EB6 !important;
    padding: 0.2rem 0.5rem !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.98rem !important;
}

/* ---------- SOURCE CITATIONS ---------- */
.sources-title {
    font-size: 0.88rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38BDF8;
    margin-top: 1.2rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 210, 255, 0.14);
    border: 1px solid rgba(0, 210, 255, 0.4);
    color: #E0F2FE !important;
    border-radius: 10px;
    padding: 0.4rem 0.9rem;
    font-size: 0.92rem;
    font-weight: 600;
    margin: 0.25rem 0.45rem 0.25rem 0;
    transition: all 0.2s ease;
    animation: fadeInSlideUp 0.3s ease-out;
}
.source-chip:hover {
    background: rgba(0, 210, 255, 0.25);
    border-color: #FF2E93;
    color: #FFFFFF !important;
    transform: translateY(-1.5px);
}
.source-page-badge {
    background: rgba(255, 46, 147, 0.3);
    color: #FFB3D9;
    font-weight: 800;
    padding: 0.12rem 0.5rem;
    border-radius: 6px;
    font-size: 0.84rem;
}

/* ---------- ANSWER VERIFICATION CARD ---------- */
.verify-container {
    background: rgba(10, 6, 18, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
}
.verify-summary-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}
.badge-high-confidence {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #34D399;
}
.badge-moderate-confidence {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #FBBF24;
}
.verify-row-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 0.96rem;
}
.verify-row-item:last-child {
    border-bottom: none;
}
.verify-sim-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    font-weight: 700;
    color: #A5F3FC;
    background: rgba(0, 210, 255, 0.12);
    border: 1px solid rgba(0, 210, 255, 0.25);
    padding: 0.18rem 0.5rem;
    border-radius: 6px;
    min-width: 50px;
    text-align: center;
}
.verify-status-supported {
    color: #34D399;
    font-weight: 700;
}
.verify-status-review {
    color: #FBBF24;
    font-weight: 700;
}

/* Expanders */
details {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    margin-top: 0.85rem !important;
}
summary {
    color: #A5F3FC !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    padding: 0.55rem 0.85rem !important;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    background: rgba(16, 10, 26, 0.94) !important;
    border-top: 1.5px solid rgba(255, 46, 147, 0.3) !important;
    backdrop-filter: blur(22px);
}
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
    color: #FFFFFF !important;
    border-radius: 16px !important;
    font-size: 1.06rem !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.35) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #00D2FF !important;
    box-shadow: 0 0 0 3px rgba(0, 210, 255, 0.3) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9D94B3 !important;
    font-size: 1.02rem !important;
}

/* Download Buttons inside Messages */
.message-actions-bar {
    display: flex;
    justify-content: flex-end;
    gap: 0.6rem;
    margin-top: 1rem;
    padding-top: 0.65rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #F1EDFC !important;
    font-size: 0.92rem !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: rgba(255, 46, 147, 0.25) !important;
    border-color: #FF2E93 !important;
    color: #FFFFFF !important;
}

/* Footer Tagline */
.app-footer {
    text-align: center;
    color: #8C84A0;
    font-size: 0.88rem;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    line-height: 1.7;
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

            # Render Source Chips (Strictly synchronized with verified supporting quotes)
            raw_sources = entry.get("sources", [])
            verification = entry.get("verification", [])
            
            supported_quotes = [
                " ".join(re.sub(r'[^\w\s]', '', r.get("quote", "").lower()).split())
                for r in verification
                if r.get("supported", False) and r.get("quote")
            ]
            
            display_sources = []
            if supported_quotes and raw_sources:
                for doc in raw_sources:
                    doc_text = (doc.get("page_content", "") if isinstance(doc, dict) else getattr(doc, "page_content", "")).lower()
                    doc_norm = " ".join(re.sub(r'[^\w\s]', '', doc_text).split())
                    if any(q in doc_norm or q[:25] in doc_norm for q in supported_quotes):
                        display_sources.append(doc)
            if not display_sources:
                display_sources = raw_sources

            if display_sources:
                seen = set()
                chips_html = ""
                source_lines_plain = []
                for doc in display_sources:
                    if isinstance(doc, dict):
                        fname = doc.get("source_file", "Document")
                        page = doc.get("page", "?")
                    else:
                        fname = doc.metadata.get("source_file", "Document")
                        page = doc.metadata.get("page", "?")
                    key = (fname, page)
                    if key not in seen:
                        chips_html += f"<span class='source-chip'>⎘ {fname} <span class='source-page-badge'>p. {page}</span></span>"
                        source_lines_plain.append(f"- {fname}, page {page}")
                        seen.add(key)

                if chips_html:
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

