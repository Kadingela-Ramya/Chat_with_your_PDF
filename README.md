# 📄 Chat With Your PDF — AI Document Intelligence Platform

**Chat With Your PDF** is a multi-document Question & Answering (Q&A) platform with exact page-level source citations and automated hallucination/grounding verification. It allows users to upload multiple PDF documents, automatically index and embed them using **Mistral AI** and **FAISS**, query across all documents simultaneously, inspect factual confidence scores, and maintain persistent user-specific chat histories.

---

## 🏗️ Architecture Overview

The system implements a Modular Retrieval-Augmented Generation (RAG) architecture:
1. **Document Ingestion & Chunking**: Uploaded PDFs are parsed page-by-page using `PyPDFLoader` and split into overlapping semantic chunks (`RecursiveCharacterTextSplitter`).
2. **Vector Indexing & Retrieval**: Chunks are transformed into dense vector embeddings using Mistral AI (`mistral-embed`) and indexed locally using **FAISS** (`IndexFlatL2`) for sub-millisecond similarity retrieval.
3. **Contextual Generation**: `ChatMistralAI` (`mistral-small-latest`) processes the top retrieved context chunks using a strict grounding prompt template to synthesize factual answers without hallucinations.
4. **Sentence Grounding & Verification Layer**: Every generated sentence is embedded and compared against retrieved PDF source chunks via Cosine Similarity (`MistralAIEmbeddings`) to compute factual support scores (`Supported` vs. `Needs Review`).
5. **Persistence & Presentation**: Multi-turn conversations, sources, and verification results are persisted in a local **SQLite** database (`chat_history.db`) and delivered through a responsive **Streamlit** interface.

---

## ⚙️ Prerequisites & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Mistral AI API Key** (Get one at [console.mistral.ai](https://console.mistral.ai/))
- **Git** (optional)

### 2. Installation Steps

1. **Clone or Open the Project Folder**:
   ```bash
   cd C:\Users\hp\Downloads\miniproject
   ```

2. **Create and Activate a Virtual Environment**:
   - **PowerShell**:
     ```powershell
     python -m venv pdf-rag-project\venv
     .\pdf-rag-project\venv\Scripts\Activate.ps1
     ```
   - **Command Prompt (CMD)**:
     ```cmd
     python -m venv pdf-rag-project\venv
     pdf-rag-project\venv\Scripts\activate.bat
     ```

3. **Install Dependencies**:
   ```bash
   pip install streamlit langchain langchain-community langchain-mistralai faiss-cpu pypdf bcrypt python-dotenv numpy
   ```

---

## 🔑 Required Environment Variables (`.env`)

Create a `.env` file in the root project directory with the following variables:

```env
# Mistral AI API Key for Embeddings and LLM Generation
MISTRAL_API_KEY=your_mistral_api_key_here

# Application Credentials (Constant-time auth check)
APP_USERNAME=admin
APP_PASSWORD_HASH=$2b$12$eXampleHashedPasswordHere...
```

### How to Generate `APP_PASSWORD_HASH`:
Run the helper utility script in your terminal:
```bash
python generate_password_hash.py
```
Type your desired password (e.g. `admin123`), press Enter, and copy the generated `APP_PASSWORD_HASH` into your `.env` file.

---

## 🚀 How to Run Locally

### Option A: Via VS Code / PowerShell
```powershell
& ".\pdf-rag-project\venv\Scripts\streamlit.exe" run app.py
```

### Option B: Via Command Prompt (CMD)
```cmd
pdf-rag-project\venv\Scripts\streamlit.exe run app.py
```

Open your browser and navigate to: **`http://localhost:8501`**

---

## 👥 Authentication & Deployment Explained

### "If I deploy the project, does everyone use `admin` / `admin123`?"

- **How it works currently**: 
  - The application is protected by the admin credentials configured in `.env` (`APP_USERNAME` and `APP_PASSWORD_HASH`).
  - This acts as an **access gate** so only authorized people with your credentials can access your deployed instance.
  - You can change `APP_USERNAME` and generate a new password anytime in `.env` before deploying.
- **Chat History Isolation**:
  - The SQLite database associates chat history with the username entered at sign-in.
  - If multiple teammates sign in using `admin`, they will see the shared `admin` chat history.
  - If you configure separate user accounts or allow custom username logins, each username maintains its own private, isolated chat history in SQLite.

---

## ⚠️ Known Limitations & Best Practices

1. **Scanned / Image-Only PDFs**:
   - `PyPDFLoader` extracts textual PDF streams. PDFs containing only scanned images (without embedded OCR text) require optical character recognition (OCR) before ingestion.
2. **Mistral API Rate Limits**:
   - Embedding large volumes of PDF pages simultaneously depends on your Mistral API tier rate limits.
3. **Session Vector Store Lifecycle**:
   - Vector indexes are built locally and held per session. When you click **Clear All Documents**, the active index and temporary cache are purged cleanly.
4. **SQLite Multi-Node Deployment**:
   - SQLite is a lightweight, zero-configuration local database ideal for single-server or single-container deployments (e.g. Streamlit Community Cloud, Render, single VM). For multi-replica distributed setups, a PostgreSQL instance is recommended.

---

## 📄 License & Attribution
- Built with **Streamlit**, **LangChain**, **FAISS**, and **Mistral AI**.
