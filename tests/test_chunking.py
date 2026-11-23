"""Tests for section-based and FAQ-aware text chunking service."""

import unittest

from app.services.chunking import chunk_document, chunk_text
from app.services.extraction import DEFAULT_DOCUMENTS_DIR, extract_document


class ChunkingTests(unittest.TestCase):
    """Verify section-based, FAQ, and sentence-overlap chunking."""

    def test_splits_by_markdown_headings(self) -> None:
        text = (
            "# Network Setup\n"
            "Connect the router to power and wait two minutes.\n\n"
            "## VPN Access\n"
            "Open the VPN client and sign in with SSO."
        )
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading, "Network Setup")
        self.assertIn("Connect the router", chunks[0].content)
        self.assertEqual(chunks[1].heading, "VPN Access")
        self.assertIn("Open the VPN client", chunks[1].content)

    def test_splits_qa_prefix_faq(self) -> None:
        text = (
            "## Account FAQ\n"
            "Q1: How do I reset my password?\n"
            "Go to the login page and click Forgot Password.\n\n"
            "Q2: Why is my account locked?\n"
            "Too many failed attempts lock the account for 15 minutes."
        )
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_type, "faq")
        self.assertEqual(chunks[0].heading, "Account FAQ")
        self.assertEqual(chunks[0].question, "Q1: How do I reset my password?")
        self.assertIn("Forgot Password", chunks[0].content)
        self.assertEqual(chunks[1].heading, "Account FAQ")
        self.assertEqual(chunks[1].question, "Q2: Why is my account locked?")

    def test_oversized_section_splits_on_sentences_with_overlap(self) -> None:
        sentences = [
            f"Sentence number {i} explains a detailed helpdesk troubleshooting step clearly."
            for i in range(1, 25)
        ]
        text = "## Long Guide\n" + " ".join(sentences)
        chunks = chunk_text(text, max_tokens=60, overlap_tokens=15)

        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk.heading, "Long Guide")
            self.assertTrue(chunk.content.startswith("## Long Guide") or chunk.content.startswith("Long Guide"))
            # Sentences should not be cut mid-sentence
            self.assertTrue(chunk.content.rstrip().endswith("."))

    def test_chunk_sample_wifi_pdf_keeps_all_20_faqs_intact(self) -> None:
        pdf_path = DEFAULT_DOCUMENTS_DIR / "wifi_troubleshooting.pdf"
        doc = extract_document(pdf_path)
        chunks = chunk_document(doc)

        self.assertEqual(len(chunks), 20)
        self.assertTrue(all(c.section_type == "faq" for c in chunks))
        # Q6 spans page 1 and page 2
        self.assertEqual(chunks[5].metadata.get("page_numbers"), [1, 2])
        # Q12 spans page 2 and page 3
        self.assertEqual(chunks[11].metadata.get("page_numbers"), [2, 3])
        # Q19 spans page 3 and page 4
        self.assertEqual(chunks[18].metadata.get("page_numbers"), [3, 4])

    def test_invalid_chunk_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("hello", max_tokens=0)
        with self.assertRaises(ValueError):
            chunk_text("hello", max_tokens=50, overlap_tokens=50)


if __name__ == "__main__":
    unittest.main()
