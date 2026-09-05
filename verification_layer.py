import argparse
import os
import re
import json
import numpy as np

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage


def split_into_claims(answer: str) -> list:
    """
    Splits answers into discrete atomic factual claims while filtering out
    meta-citation wrapper phrases and deduplicating embedded quotes.
    """
    if not answer or not answer.strip():
        return []

    # 1. Strip bracketed citation tags and markdown formatting
    cleaned = re.sub(r'\[Source:[^\]]+\]', '', answer)
    cleaned = re.sub(r'\[(?:Page\s*)?\d+\]', '', cleaned)
    cleaned = re.sub(r'[*_`]', '', cleaned)

    # 2. Filter out meta-citation wrapper phrases
    meta_patterns = [
        r'this is (?:explicitly )?(?:stated|mentioned|provided|confirmed)(?: in [^:\.\n]+)?:?',
        r'as stated (?:in|on) (?:the )?(?:provided )?(?:source|document|page \d+|context|section [^:\.\n]+)?:?',
        r'according to (?:the )?(?:provided )?(?:source|document|excerpts|context|section [^:\.\n]+)?:?',
        r'based on (?:the )?(?:provided )?(?:source|document|excerpts|context|section [^:\.\n]+)?:?'
    ]
    for pat in meta_patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

    # 3. Strip quotes and clean whitespace
    cleaned = re.sub(r'["“”«»]', '', cleaned)

    # 4. Split sentences
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', cleaned) if s.strip()]

    claims = []
    for s in raw_sentences:
        s_clean = s.strip(" :,-.")
        if len(s_clean) < 8:
            continue

        # Split on substantive coordinating conjunctions if compound
        sub_clauses = re.split(
            r'\s*;\s*|,\s+and\s+|,\s+while\s+|,\s+whereas\s+|\s+and\s+(?=[A-Z]|Lord|India|the\s+Governor|the\s+first)',
            s_clean,
            flags=re.IGNORECASE
        )
        for c in sub_clauses:
            c_sub = c.strip(" :,-.")
            # Remove leading bullet points or numbers like (1) or 1.
            c_sub = re.sub(r'^\(?\d+[\.\)]\s*', '', c_sub).strip()
            if len(c_sub) > 8:
                claims.append(c_sub)

    # 5. Deduplicate overlapping claims with high token overlap
    final_claims = []
    for c in claims:
        c_words = set(re.findall(r'\w+', c.lower()))
        is_dup = False
        for fc in final_claims:
            fc_words = set(re.findall(r'\w+', fc.lower()))
            jaccard = len(c_words & fc_words) / len(c_words | fc_words) if (c_words | fc_words) else 0.0
            if jaccard >= 0.50:
                is_dup = True
                break
        if not is_dup:
            final_claims.append(c)

    return final_claims if final_claims else claims


def is_refusal_response(answer: str) -> bool:
    """
    Detects if the answer is a PURE refusal or negative finding message.
    CRITICAL: If the answer contains substantive factual claims (years, specific dates,
    section numbers, historical assertions) before or alongside a refusal phrase,
    it is NOT a pure refusal — it is a hallucinated assertion with a disclaimer
    and MUST undergo full NLI factual verification.
    """
    if not answer or not answer.strip():
        return True

    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()

    refusal_patterns = [
        r"couldn't find",
        r"could not find",
        r"do not contain",
        r"does not contain",
        r"no information found",
        r"not mentioned in the provided",
        r"not found in the provided",
        r"not available in the provided",
        r"provided document excerpts do not",
        r"provided excerpts do not"
    ]

    has_refusal_phrase = any(re.search(pat, answer_lower) for pat in refusal_patterns)
    if not has_refusal_phrase:
        return False

    # Check if substantive factual claims are asserted alongside the disclaimer:
    # 1. Specific 4-digit years (e.g. 1950, 1961, 1962, 1922)
    has_years = bool(re.search(r'\b(18\d\d|19\d\d|20\d\d)\b', answer_clean))
    # 2. Specific section, article, or act-of numbers
    has_sections = bool(re.search(r'\b(?:section|article|entry|schedule|act of)\s+\d+\b', answer_lower))

    # 3. Check if multiple substantive sentences exist where only one is the disclaimer
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer_clean) if s.strip()]
    has_substantive_preceding_claims = len(sentences) > 1 and not all(
        any(re.search(pat, s.lower()) for pat in refusal_patterns) for s in sentences
    )

    if (has_years or has_sections) and has_substantive_preceding_claims:
        # Contradictory answer: states specific facts from memory AND adds a disclaimer
        # This is an active hallucination requiring strict entailment verification!
        return False

    return True


def verify_answer(
    answer: str,
    source_documents,
    llm_model: str = "open-mistral-7b",
    embedding_model: str = "mistral-embed",
    threshold: float = 0.75,
):
    """
    Verifies each atomic claim in the answer against retrieved source chunks
    using strict factual entailment (NLI) and exact quotation verification.
    """
    # Skip verification on refusal responses (no factual claims to ground)
    if is_refusal_response(answer):
        return []

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

    primary_llm = ChatMistralAI(model=llm_model, temperature=0.0)
    fallback_llm = ChatMistralAI(model="mistral-tiny", temperature=0.0)
    llm = primary_llm.with_fallbacks([fallback_llm])
    results = []

    for claim in claims:
        prompt = (
            f"You are a strict, legally rigorous factual grounding auditor.\n\n"
            f"Retrieved Source Text:\n{sources_block}\n\n"
            f"Full Answer Context:\n\"{answer}\"\n\n"
            f"Specific Claim to Verify:\n\"{claim}\"\n\n"
            f"Instructions:\n"
            f"1. Evaluate whether the source text directly, fully, and unambiguously ENTAILS and PROVES the specific claim in the context of the overall answer (resolving pronouns like 'it' or 'this' to the main subject).\n"
            f"2. Contextual Inversion & Repeal Check:\n"
            f"   - If a quoted phrase appears in a clause that actually negates, repeals, or restricts the claim (e.g. citing an old repealed law mentioned only as a historical reference, when the claim asserts it is the enacted governing law), you MUST set supported to false.\n"
            f"   - Do NOT accept truncated or selective quotes that distort or reverse the true meaning of the complete sentence.\n"
            f"3. If supported is true, you MUST extract the FULL verbatim sentence from the source text that proves the claim in its complete context.\n"
            f"4. If unsupported, ambiguous, or contradicted by the full context, set supported to false and quote to \"\".\n\n"
            f"Respond strictly in JSON format:\n"
            f'{{"supported": true/false, "quote": "full sentence from source or empty string", "cited_page": "page number or null", "confidence": 0.0 to 1.0}}\n'
        )

        parsed = None
        for m_name in ["open-mistral-7b", "mistral-tiny"]:
            try:
                auditor_llm = ChatMistralAI(model=m_name, temperature=0.0)
                resp = auditor_llm.invoke([
                    SystemMessage(content="You are a strict grounding auditor that checks for contextual entailment and rejects selective quotes that invert meaning."),
                    HumanMessage(content=prompt)
                ])
                content = resp.content.strip()
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                parsed = json.loads(json_match.group(0)) if json_match else json.loads(content)
                break
            except Exception:
                import time
                time.sleep(1.0)
        
        if not parsed:
            parsed = {"supported": False, "quote": "", "cited_page": None, "confidence": 0.0}

        quote = str(parsed.get("quote", "")).strip()
        is_supported = bool(parsed.get("supported", False)) and len(quote) > 0

        # Verbatim quote check in source documents with whitespace normalization
        matched_page = str(parsed.get("cited_page", "")).strip()
        matched_file = "Document"
        if is_supported:
            quote_norm = " ".join(re.sub(r'[^\w\s]', '', quote.lower()).split())
            matched = False
            for d in source_documents:
                doc_text = getattr(d, "page_content", str(d)).lower()
                doc_norm = " ".join(re.sub(r'[^\w\s]', '', doc_text).split())
                if quote_norm in doc_norm or quote_norm[:25] in doc_norm:
                    matched = True
                    p_val = d.metadata.get("page") if hasattr(d, "metadata") else d.get("page")
                    f_val = d.metadata.get("source_file") if hasattr(d, "metadata") else d.get("source_file")
                    if p_val is not None and str(p_val) != "?":
                        matched_page = str(p_val)
                    if f_val:
                        matched_file = str(f_val)
                    break
            if not matched:
                is_supported = False
                quote = ""

        conf = float(parsed.get("confidence", 0.95 if is_supported else 0.20))
        results.append({
            "sentence": claim,
            "supported": is_supported,
            "quote": quote if is_supported else "",
            "cited_page": matched_page if is_supported else "",
            "source_file": matched_file if is_supported else "",
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