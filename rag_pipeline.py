#Execution command: python file_name.py --pdf "Path_of_the_PDF"

import argparse
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()


class PDFRAGPipelineMistral:

    def __init__(
        self,
        pdf_path: str,
        llm_model: str = "mistral-small-latest",
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

        self.vectorstore = None
        self.qa_chain = None

    def load_and_split(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        print(f"[1/4] Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print(
            f"[2/4] Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        print(f"      -> {len(documents)} pages -> {len(chunks)} chunks")
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

    def build_qa_chain(self, k: int = 4):
        print(
            f"[4/4] Building QA chain with LLM '{self.llm_model}' (top-{k} retrieval)")

        llm = ChatMistralAI(model=self.llm_model, temperature=0.1)

        prompt = PromptTemplate(
            template=(
                "Answer the question using ONLY the context below. "
                "If the answer isn't in the context, say you don't know.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
            input_variables=["context", "question"],
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return self.qa_chain

    def setup(self, force_rebuild: bool = False, k: int = 4):
        chunks = self.load_and_split()
        self.build_vectorstore(chunks, force_rebuild=force_rebuild)
        self.build_qa_chain(k=k)

    def ask(self, question: str):
        if self.qa_chain is None:
            raise RuntimeError("Call setup() before ask().")
        result = self.qa_chain.invoke({"query": question})
        return result["result"], result.get("source_documents", [])


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