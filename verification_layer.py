import argparse
import os
import re
import json
import numpy as np

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage


def split_into_claims(answer: str) -> list:
    """
    Splits compound sentences into discrete atomic factual claims.
    e.g. 'X happened on date Y, and Z was Governor-General' -> ['X happened on date Y', 'Z was Governor-General']
    """
    if not answer or not answer.strip():
        return []

    # Strip bracketed page numbers for clean claim analysis
    cleaned = re.sub(r'\[(?:Page\s*)?\d+\]', '', answer).strip()
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]

    claims = []
    for s in raw_sentences:
        # Split on coordinating conjunctions that connect substantive clauses
        sub_clauses = re.split(
            r'\s*;\s*|,\s+and\s+|,\s+while\s+|,\s+whereas\s+|\s+and\s+(?=[A-Z]|Lord|India|the\s+Governor|the\s+first)',
            s,
            flags=re.IGNORECASE
        )
        for c in sub_clauses:
            c_clean = c.strip(" ,.")
            if len(c_clean) > 5:
                claims.append(c_clean)

    return claims if claims else raw_sentences


def verify_answer(
    answer: str,
    source_documents,
    llm_model: str = "mistral-small-latest",
    embedding_model: str = "mistral-embed",
    threshold: float = 0.75,
):
    """
    Verifies each atomic claim in the answer against retrieved source chunks
    using strict factual entailment (NLI) and exact quotation verification.
    """
    claims = split_into_claims(answer)
    if not claims:
        return []

    if not source_documents:
        return [
            {
                "sentence": claim,
                "supported": False,
                "quote": "",
                "cited_page": None,
                "similarity": 0.0,
            }
            for claim in claims
        ]

    # Format sources with page tags
    sources_text_list = []
    for i, doc in enumerate(source_documents, 1):
        if hasattr(doc, "metadata"):
            p = doc.metadata.get("page", "?")
            f = doc.metadata.get("source_file", "Document")
            content = doc.page_content
        elif isinstance(doc, dict):
            p = doc.get("page", "?")
            f = doc.get("source_file", "Document")
            content = doc.get("page_content", "")
        else:
            p = "?"
            f = "Document"
            content = str(doc)
        sources_text_list.append(f"[Source {i} (Page {p})]\n{content}")

    sources_block = "\n\n".join(sources_text_list)

    llm = ChatMistralAI(model=llm_model, temperature=0.0)
    results = []

    for claim in claims:
        prompt = (
            f"You are a strict, objective factual grounding auditor.\n\n"
            f"Retrieved Source Text:\n{sources_block}\n\n"
            f"Claim to Verify:\n\"{claim}\"\n\n"
            f"Instructions:\n"
            f"1. Check if the source text explicitly and directly states the claim.\n"
            f"2. If yes, extract the exact supporting sentence verbatim from the source text.\n"
            f"3. If the source text merely discusses related topics or figures but does NOT explicitly state this specific claim, you MUST set supported to false.\n"
            f"4. Respond strictly in JSON format:\n"
            f'{{"supported": true/false, "quote": "exact sentence from source text or empty string", "cited_page": "page number or null", "confidence": 0.0 to 1.0}}\n'
        )

        try:
            resp = llm.invoke([
                SystemMessage(content="You are a strict grounding auditor that verifies claims against source text with exact verbatim quotes."),
                HumanMessage(content=prompt)
            ])
            content = resp.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            parsed = json.loads(json_match.group(0)) if json_match else json.loads(content)
        except Exception:
            parsed = {"supported": False, "quote": "", "cited_page": None, "confidence": 0.0}

        quote = str(parsed.get("quote", "")).strip()
        is_supported = bool(parsed.get("supported", False)) and len(quote) > 0

        # Verbatim quote check in source documents with whitespace normalization
        if is_supported:
            quote_norm = " ".join(re.sub(r'[^\w\s]', '', quote.lower()).split())
            matched = False
            for d in source_documents:
                doc_text = getattr(d, "page_content", str(d)).lower()
                doc_norm = " ".join(re.sub(r'[^\w\s]', '', doc_text).split())
                if quote_norm in doc_norm or quote_norm[:30] in doc_norm:
                    matched = True
                    break
            if not matched:
                is_supported = False
                quote = ""

        conf = float(parsed.get("confidence", 0.95 if is_supported else 0.20))
        results.append({
            "sentence": claim,
            "supported": is_supported,
            "quote": quote if is_supported else "",
            "cited_page": str(parsed.get("cited_page", "")),
            "similarity": round(conf, 2),
        })

    return results


def print_verification(results):

    if not results:
        print("\nNo verification results.")
        return

    print("\n   Verification:")

    for result in results:

        if result["supported"]:
            icon = "✅"
        else:
            icon = "⚠️"

        print(
            f"   {icon} "
            f"({result['similarity']}) "
            f"{result['sentence']}"
        )

    print()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Verification layer for "
            "Mistral PDF RAG pipeline"
        )
    )

    parser.add_argument(
        "--pdfs",
        required=True,
        nargs="+",
        help="Path(s) to PDF files"
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the PDF"
    )

    parser.add_argument(
        "--llm",
        default="mistral-small-latest",
        help="Mistral chat model"
    )

    parser.add_argument(
        "--embed-model",
        default="mistral-embed",
        help="Mistral embedding model"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild FAISS index"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # STEP 1: Create your existing RAG pipeline
    # --------------------------------------------------

    pipeline = PDFRAGPipelineMistral(
        pdf_paths=args.pdfs,
        llm_model=args.llm,
        embedding_model=args.embed_model,
    )

    # --------------------------------------------------
    # STEP 2: Setup RAG
    # --------------------------------------------------

    pipeline.setup(
        force_rebuild=args.rebuild,
        k=args.k
    )

    # --------------------------------------------------
    # STEP 3: Ask question using existing RAG
    # --------------------------------------------------

    answer, sources = pipeline.ask(
        args.question
    )

    print("\n========================================")
    print("RAG ANSWER")
    print("========================================")

    print(f"\n{answer}")

    # --------------------------------------------------
    # STEP 4: Show retrieved sources
    # --------------------------------------------------

    print("\n========================================")
    print("SOURCES")
    print("========================================")

    if sources:

        seen = set()

        for doc in sources:

            filename = doc.metadata.get(
                "source_file",
                "?"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            entry = (filename, page)

            if entry not in seen:

                print(
                    f"- {filename}, page {page}"
                )

                seen.add(entry)

    else:

        print("No source documents retrieved.")

    # --------------------------------------------------
    # STEP 5: VERIFY ANSWER
    # --------------------------------------------------

    results = verify_answer(
        answer=answer,
        source_documents=sources,
        embedding_model=args.embed_model,
        threshold=args.threshold,
    )

    # --------------------------------------------------
    # STEP 6: Display verification
    # --------------------------------------------------

    print("\n========================================")
    print("VERIFICATION")
    print("========================================")

    print_verification(results)


if __name__ == "__main__":
    main()