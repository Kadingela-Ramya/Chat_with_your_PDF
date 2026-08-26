#!/usr/bin/env python3
"""
=============================================================================
🛡️ Automated Factual Grounding & Retrieval Verification Test Suite
=============================================================================
Runs a standard 5-probe test checklist against any PDF (or multiple PDFs)
to evaluate factual accuracy, citation integrity, parametric leakage,
absurdity refusal, and contextual inversion.

Usage:
    python test_grounding.py --pdf "path/to/document.pdf"
    python test_grounding.py --pdfs "doc1.pdf" "doc2.pdf" --output report.md
=============================================================================
"""

import os
import sys
import re
import json
import argparse
import time
from datetime import datetime

# UTF-8 stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline_v2 import PDFRAGPipelineMistral
from verification_layer import verify_answer, is_refusal_response


class GroundingTestSuite:
    def __init__(self, pdf_paths: list, persist_dir: str = "test_faiss_index"):
        self.pdf_paths = [os.path.abspath(p) for p in pdf_paths]
        self.persist_dir = persist_dir
        self.pipeline = None
        self.test_results = []

    def setup(self):
        print(f"\n========================================================")
        print(f"🚀 INITIALIZING GROUNDING TEST SUITE")
        print(f"   Documents ({len(self.pdf_paths)}): {[os.path.basename(p) for p in self.pdf_paths]}")
        print(f"========================================================")
        
        self.pipeline = PDFRAGPipelineMistral(
            pdf_paths=self.pdf_paths,
            persist_dir=self.persist_dir,
            chunk_size=1000,
            chunk_overlap=150,
        )
        self.pipeline.setup(force_rebuild=True, k=8)
        print(f"✔ Pipeline successfully initialized with Hybrid Retriever (BM25 + FAISS).\n")

    def run_probe(self, probe_id: int, probe_name: str, probe_type: str, question: str, expected_behavior: str):
        print(f"\n--------------------------------------------------------")
        print(f"🧪 [PROBE {probe_id}/5]: {probe_name}")
        print(f"   Type: {probe_type}")
        print(f"   Question: \"{question}\"")
        print(f"   Expected: {expected_behavior}")
        print(f"--------------------------------------------------------")
        
        start_time = time.time()
        answer, raw_sources = self.pipeline.ask(question)
        latency = time.time() - start_time
        
        # 1. Physical citation presence check
        valid_retrieved_pages = {
            str(d.metadata.get("page")) for d in raw_sources
            if hasattr(d, "metadata") and d.metadata.get("page") is not None
        }
        
        cited_pages = set(re.findall(r'\[(?:Page\s*)?(\d+)\]', answer, re.IGNORECASE))
        fabricated_pages = cited_pages - valid_retrieved_pages
        
        # 2. Multi-document fanout check
        retrieved_files = {
            os.path.basename(d.metadata.get("source_file", ""))
            for d in raw_sources if hasattr(d, "metadata")
        }
        
        # 3. Refusal detection
        is_refusal = is_refusal_response(answer)
        
        # 4. NLI Factual Grounding Verification
        if is_refusal and not any(kw in answer.lower() for kw in ["page", "article", "section", "act"]):
            v_report = []
            grounding_percent = 100 if probe_type in ["Absurdity Probe", "Negative / Out-of-Scope Test"] else 0
            status_flag = "PASS (Clean Refusal)"
        else:
            v_report = verify_answer(answer, raw_sources)
            if v_report:
                sup_count = sum(1 for r in v_report if r.get("supported", False))
                grounding_percent = int((sup_count / len(v_report)) * 100)
                status_flag = "PASS" if grounding_percent == 100 and not fabricated_pages else "NEEDS REVIEW"
            else:
                grounding_percent = 0
                status_flag = "NEEDS REVIEW (Empty Verification)"

        # Print Live Terminal Summary
        print(f"\n📄 GENERATED ANSWER:")
        print(f"   {answer}\n")
        
        print(f"⎘ RETRIEVED SOURCES ({len(raw_sources)} chunks across {len(retrieved_files)} files):")
        for s in raw_sources:
            f = s.metadata.get("source_file")
            p = s.metadata.get("page")
            print(f"   • {f} | Page {p}")
            
        print(f"\n🛡️ GROUNDING AUDIT REPORT:")
        if v_report:
            for r in v_report:
                mark = "✓ Supported" if r.get("supported") else "⚠ Unsupported"
                print(f"   [{mark}] Claim: \"{r.get('sentence')}\"")
                if r.get("supported"):
                    print(f"      💬 Quote: \"{r.get('quote')}\" (Page {r.get('cited_page')})")
                else:
                    print(f"      ⚠ Reason: No direct supporting statement found in retrieved sources.")
        else:
            print(f"   ℹ️ Refusal detected — verification bypassed cleanly.")
            
        print(f"\n📊 AUDIT VERDICT: {status_flag} | Grounding Score: {grounding_percent}% | Latency: {latency:.2f}s")
        if fabricated_pages:
            print(f"   🚨 WARNING: Fabricated citations detected: {fabricated_pages}")

        probe_result = {
            "probe_id": probe_id,
            "name": probe_name,
            "type": probe_type,
            "question": question,
            "expected": expected_behavior,
            "answer": answer,
            "latency": latency,
            "sources": [{"file": s.metadata.get("source_file"), "page": s.metadata.get("page")} for s in raw_sources],
            "retrieved_files": list(retrieved_files),
            "cited_pages": list(cited_pages),
            "fabricated_pages": list(fabricated_pages),
            "is_refusal": is_refusal,
            "grounding_score": grounding_percent,
            "verification_report": v_report,
            "status": status_flag
        }
        self.test_results.append(probe_result)
        return probe_result

    def export_report(self, output_path: str = "grounding_test_report.md"):
        print(f"\n========================================================")
        print(f"💾 EXPORTING AUDIT REPORT -> {output_path}")
        print(f"========================================================")
        
        md = []
        md.append(f"# 🛡️ Grounding & Retrieval Verification Audit Report")
        md.append(f"- **Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"- **Audited Documents:** {', '.join([os.path.basename(p) for p in self.pdf_paths])}")
        md.append(f"- **Total Probes Executed:** {len(self.test_results)}\n")
        
        # Summary Table
        md.append(f"## 📊 Executive Summary Table")
        md.append(f"| Probe | Name | Type | Grounding % | Fabricated Citations | Verdict |")
        md.append(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for r in self.test_results:
            fab_str = "None" if not r["fabricated_pages"] else f"🚨 {r['fabricated_pages']}"
            md.append(f"| #{r['probe_id']} | **{r['name']}** | {r['type']} | {r['grounding_score']}% | {fab_str} | **{r['status']}** |")
            
        md.append(f"\n---\n")
        
        # Detailed Probe Breakdown
        md.append(f"## 🔬 Detailed Probe-by-Probe Breakdown\n")
        for r in self.test_results:
            md.append(f"### Probe #{r['probe_id']}: {r['name']} ({r['type']})")
            md.append(f"- **Question:** *\"{r['question']}\"*")
            md.append(f"- **Expected Behavior:** {r['expected']}")
            md.append(f"- **Latency:** {r['latency']:.2f} seconds")
            md.append(f"- **Status Verdict:** `{r['status']}`\n")
            
            md.append(f"#### Generated Answer:")
            md.append(f"> {r['answer']}\n")
            
            md.append(f"#### Retrieved Sources:")
            for s in r["sources"]:
                md.append(f"- `⎘ {s.get('file')} | Page {s.get('page')}`")
            md.append("")
            
            md.append(f"#### Grounding & Verification Audit:")
            if r["verification_report"]:
                for v in r["verification_report"]:
                    sup = "✓ Supported" if v.get("supported") else "⚠ Unsupported"
                    md.append(f"- **[{sup}]** `{v.get('sentence')}`")
                    if v.get("supported"):
                        md.append(f"  - 💬 *Source Quote:* \"{v.get('quote')}\" (Page {v.get('cited_page')})")
                    else:
                        md.append(f"  - ⚠ *Reason:* No direct supporting statement found in retrieved chunks.")
            else:
                md.append(f"- *Clean refusal detected — verification bypassed.*")
            md.append(f"\n---\n")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
            
        print(f"✔ Audit report successfully saved to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Automated Grounding Test Suite for PDFs")
    parser.add_argument("--pdf", type=str, help="Path to single PDF")
    parser.add_argument("--pdfs", nargs="+", help="Paths to multiple PDFs")
    parser.add_argument("--output", type=str, default="grounding_test_report.md", help="Output markdown report path")
    
    # Custom Questions Override
    parser.add_argument("--q1", type=str, help="Probe 1: Easy factual check question")
    parser.add_argument("--q2", type=str, help="Probe 2: Absurdity probe question")
    parser.add_argument("--q3", type=str, help="Probe 3: Negative/out-of-scope question")
    parser.add_argument("--q4", type=str, help="Probe 4: Cross-document synthesis question")
    parser.add_argument("--q5", type=str, help="Probe 5: Quote-context/inversion check question")

    args = parser.parse_args()
    
    pdf_list = []
    if args.pdfs:
        pdf_list = args.pdfs
    elif args.pdf:
        pdf_list = [args.pdf]
    else:
        # Default to existing project PDFs if available
        default_files = [
            r"C:\Users\hp\Downloads\Constitution_of_India402.pdf",
            r"C:\Users\hp\Downloads\IncomeTax_File686.pdf"
        ]
        pdf_list = [f for f in default_files if os.path.exists(f)]
        if not pdf_list:
            print("❌ Error: Please provide at least one PDF path using --pdf or --pdfs.")
            sys.exit(1)

    suite = GroundingTestSuite(pdf_paths=pdf_list)
    suite.setup()

    # Standard Probes
    q1 = args.q1 or "What is the short title of the Income-tax Act and when does it come into force?"
    q2 = args.q2 or "What was the name of the supersonic jet used during the constituent assembly meetings?"
    q3 = args.q3 or "Which section specifies the tax rate for lunar real estate transactions?"
    q4 = args.q4 or "Compare the year the Constitution was adopted with the year the Income Tax Act was enacted"
    q5 = args.q5 or "Is the Income-tax Act, 1961 currently in force or was it repealed by this Act?"

    # Execute 5 Probes
    suite.run_probe(
        1, "Easy Factual Check", "Direct Extraction",
        q1, "Extract exact title and commencement with 1:1 page citation."
    )
    suite.run_probe(
        2, "Absurdity Probe", "Adversarial Refusal",
        q2, "Refuse cleanly without fabricating anachronistic facts."
    )
    suite.run_probe(
        3, "Negative / Out-of-Scope Test", "Parametric Boundary",
        q3, "Refuse cleanly rather than answering from external knowledge."
    )
    suite.run_probe(
        4, "Cross-Document Synthesis", "Multi-Document Retrieval",
        q4, "Fan out across all loaded PDFs; declare missing sections honestly."
    )
    suite.run_probe(
        5, "Quote-Context & Inversion Check", "Semantic Entailment",
        q5, "Identify repeal clauses accurately without selective quote distortion."
    )

    # Export report
    suite.export_report(args.output)


if __name__ == "__main__":
    main()
