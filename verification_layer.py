import argparse
import numpy as np

from langchain_mistralai import MistralAIEmbeddings

from rag_pipeline_v2 import PDFRAGPipelineMistral


def verify_answer(
    answer: str,
    source_documents,
    embedding_model: str = "mistral-embed",
    threshold: float = 0.75,
):
    """
    Verify each sentence in the generated answer
    against the retrieved PDF source chunks.
    """

    embeddings_model = MistralAIEmbeddings(
        model=embedding_model
    )

    # Split answer into individual sentences
    sentences = [
        s.strip()
        for s in answer.replace("\n", " ").split(".")
        if s.strip()
    ]

    if not sentences:
        return []

    # Get text from retrieved PDF chunks
    source_texts = [
        doc.page_content
        for doc in source_documents
    ]

    # No source documents means nothing can be verified
    if not source_texts:
        return [
            {
                "sentence": sentence,
                "supported": False,
                "similarity": 0.0,
            }
            for sentence in sentences
        ]

    # Convert answer sentences into embeddings
    sentence_embeddings = (
        embeddings_model.embed_documents(sentences)
    )

    # Convert retrieved source chunks into embeddings
    source_embeddings = (
        embeddings_model.embed_documents(source_texts)
    )

    results = []

    # Compare every answer sentence
    # against every retrieved source chunk
    for sentence, sentence_embedding in zip(
        sentences,
        sentence_embeddings
    ):

        similarities = []

        for source_embedding in source_embeddings:

            # Cosine similarity
            similarity = np.dot(
                sentence_embedding,
                source_embedding
            ) / (
                np.linalg.norm(sentence_embedding)
                *
                np.linalg.norm(source_embedding)
            )

            similarities.append(similarity)

        # Take the strongest matching source
        max_similarity = max(similarities)

        results.append(
            {
                "sentence": sentence,
                "supported": (
                    max_similarity >= threshold
                ),
                "similarity": round(
                    float(max_similarity),
                    2
                ),
            }
        )

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