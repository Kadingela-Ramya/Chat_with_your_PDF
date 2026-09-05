#Execution command: python file_name.py --pdf "Path_of_the_PDF"

import argparse
import os
import re
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()


class HybridRetriever(BaseRetriever):
    bm25: BM25Retriever = None
    vectorstore: FAISS
    k: int = 8

    def _get_relevant_documents(self, query: str):
        dense_docs = self.vectorstore.similarity_search(query, k=self.k)
        if not self.bm25:
            return dense_docs

        try:
            bm25_docs = self.bm25.invoke(query)
        except Exception:
            bm25_docs = []

        if not bm25_docs:
            return dense_docs

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}

        def add_docs(docs, weight=1.0):
            for rank, doc in enumerate(docs):
                key = (doc.metadata.get("source_file"), str(doc.metadata.get("page")), doc.page_content[:60])
                if key not in doc_map:
                    doc_map[key] = doc
                    rrf_scores[key] = 0.0
                rrf_scores[key] += weight * (1.0 / (60 + rank))

        add_docs(dense_docs, weight=1.0)
        add_docs(bm25_docs, weight=1.0)

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]


class PDFRAGPipelineMistral:

    def __init__(
        self,
        pdf_path: str,
        llm_model: str = "open-mistral-7b",
        embedding_model: str = "mistral-embed",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        persist_dir: str = "faiss_index_mistral",
    ):
        self.pdf_path = pdf_path
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir

        self.chunks = []
        self.vectorstore = None
        self.qa_chain = None

    def load_and_split(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        print(f"[1/4] Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        source_name = os.path.basename(self.pdf_path)
        for doc in documents:
            doc.metadata["source_file"] = source_name
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1

        print(
            f"[2/4] Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        print(f"      -> {len(documents)} pages -> {len(chunks)} chunks")
        self.chunks = chunks
        return chunks

    def build_vectorstore(self, chunks, force_rebuild: bool = False):
        embeddings = MistralAIEmbeddings(model=self.embedding_model)

        if os.path.exists(self.persist_dir) and not force_rebuild:
            print(
                f"[3/4] Loading existing FAISS index from '{self.persist_dir}'")
            self.vectorstore = FAISS.load_local(
                self.persist_dir, embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(
                f"[3/4] Embedding {len(chunks)} chunks via Mistral API and building FAISS index")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self.vectorstore.save_local(self.persist_dir)
            print(f"      -> Index saved to '{self.persist_dir}'")

        return self.vectorstore

    def build_qa_chain(self, k: int = 8):
        print(
            f"[4/4] Building QA chain with LLM '{self.llm_model}' (top-{k} hybrid retrieval)")

        primary_llm = ChatMistralAI(model=self.llm_model, temperature=0.0)
        fallback_llm = ChatMistralAI(model="mistral-tiny", temperature=0.0)
        llm = primary_llm.with_fallbacks([fallback_llm])

        document_prompt = PromptTemplate(
            template="[Source: {source_file}, Page {page}]\n{page_content}",
            input_variables=["source_file", "page", "page_content"]
        )

        prompt = PromptTemplate(
            template=(
                "Answer the question using only the provided source chunks below.\n\n"
                "CRITICAL CITATION & GROUNDING RULES:\n"
                "1. Every claim in your answer must be directly and specifically supported by the retrieved chunks below.\n"
                "2. You may ONLY cite page numbers that literally appear in the '[Source: ..., Page X]' headers of the Context below. Never cite or invent any page number not explicitly listed in the Context headers.\n"
                "3. If the provided chunks for any document or entity do not contain the specific fact, year, article number, or section asked for, explicitly state that the provided document excerpts do not contain that information.\n"
                "4. NEVER answer from parametric training memory, extrapolate, or guess page numbers when facts are missing from the context.\n"
                "5. If you state that the provided document excerpts do not contain the information, you MUST NOT state unverified facts, years, or historical dates from memory. State ONLY that the excerpts do not contain the requested information.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        bm25 = None
        if self.chunks:
            try:
                bm25 = BM25Retriever.from_documents(self.chunks, k=k)
            except Exception:
                bm25 = None

        source_names = [os.path.basename(self.pdf_path)] if hasattr(self, "pdf_path") else []
        retriever = HybridRetriever(
            bm25=bm25,
            vectorstore=self.vectorstore,
            k=k,
            source_files=source_names
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "document_prompt": document_prompt
            },
            return_source_documents=True,
        )
        return self.qa_chain

    def setup(self, force_rebuild: bool = False, k: int = 8):
        chunks = self.load_and_split()
        self.build_vectorstore(chunks, force_rebuild=force_rebuild)
        self.build_qa_chain(k=k)

    def ask(self, question: str):
        if self.qa_chain is None:
            raise RuntimeError("Call setup() before ask().")
        result = self.qa_chain.invoke({"query": question})
        raw_answer = result["result"]
        raw_sources = result.get("source_documents", [])

        # Strict citation validation:
        # Check every cited [Page X] against pages physically present in raw_sources
        valid_retrieved_pages = {
            str(d.metadata.get("page")) for d in raw_sources
            if hasattr(d, "metadata") and d.metadata.get("page") is not None
        }

        # Any citation [Page X] where X is NOT in retrieved context was fabricated by LLM memory
        def sanitize_citations(match):
            p_num = match.group(1)
            if p_num in valid_retrieved_pages:
                return f"[Page {p_num}]"
            else:
                # Strip fabricated citation so false page numbers never reach the user
                return ""

        cleaned_answer = re.sub(r'\[(?:Page\s*)?(\d+)\]', sanitize_citations, raw_answer)
        cleaned_answer = re.sub(r'\s{2,}', ' ', cleaned_answer).strip()

        # Filter sources to only include chunks whose page numbers are legitimately cited in the answer
        cited_page_strs = set(re.findall(r'\[(?:Page\s*)?(\d+)\]', cleaned_answer, re.IGNORECASE))
        if cited_page_strs:
            filtered_sources = [
                d for d in raw_sources
                if str(d.metadata.get("page")) in cited_page_strs
            ]
            if filtered_sources:
                return cleaned_answer, filtered_sources

        return cleaned_answer, raw_sources


def main():
    parser = argparse.ArgumentParser(
        description="RAG over a PDF using the Mistral cloud API.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--llm", default="mistral-small-latest", help="Mistral chat model")
    parser.add_argument("--embed-model", default="mistral-embed",
                        help="Mistral embedding model")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the vector index")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of chunks to retrieve per question")
    parser.add_argument("--question", default=None,
                        help="Ask one question and exit (non-interactive)")
    args = parser.parse_args()

    pipeline = PDFRAGPipelineMistral(
        pdf_path=args.pdf,
        llm_model=args.llm,
        embedding_model=args.embed_model,
    )
    pipeline.setup(force_rebuild=args.rebuild, k=args.k)

    if args.question:
        answer, sources = pipeline.ask(args.question)
        print(f"\nA: {answer}\n")
        _print_sources(sources)
        return

    print("\nReady. Ask questions about the PDF (type 'exit' to quit).\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer, sources = pipeline.ask(question)
        print(f"\nA: {answer}\n")
        _print_sources(sources)


def _print_sources(sources):
    if sources:
        pages = sorted({doc.metadata.get("page", "?") for doc in sources})
        print(f"   (sources: page(s) {pages})\n")


if __name__ == "__main__":
    main()