"""Tests for document text extraction service."""

from pathlib import Path
import tempfile
import unittest

from app.services.extraction import (
    DEFAULT_DOCUMENTS_DIR,
    clean_text,
    extract_document,
    extract_text,
    list_knowledge_base_files,
)


class ExtractionTests(unittest.TestCase):
    """Verify text extraction across supported document formats."""

    def test_clean_text_normalizes_whitespace(self) -> None:
        raw = "Line   one\r\n\r\n\r\n\tLine\t\ttwo   \n"
        self.assertEqual(clean_text(raw), "Line one\n\nLine two")

    def test_extract_txt_and_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            txt_file = tmp / "guide.txt"
            md_file = tmp / "faq.md"
            bad_file = tmp / "notes.csv"

            txt_file.write_text("Hello   world\r\n\r\nSecond paragraph", encoding="utf-8")
            md_file.write_text("# Title\n\nSome   markdown", encoding="utf-8")
            bad_file.write_text("a,b\n1,2", encoding="utf-8")

            txt_doc = extract_document(txt_file)
            self.assertEqual(txt_doc.text, "Hello world\n\nSecond paragraph")
            self.assertEqual(txt_doc.metadata["file_type"], "txt")
            self.assertEqual(txt_doc.metadata["page_count"], 1)

            self.assertEqual(extract_text(md_file), "# Title\n\nSome markdown")

            with self.assertRaises(ValueError):
                extract_document(bad_file)

    def test_extract_sample_pdf(self) -> None:
        files = list_knowledge_base_files(DEFAULT_DOCUMENTS_DIR)
        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        self.assertTrue(pdf_files, "Expected at least one PDF in knowledge_base/documents")

        doc = extract_document(pdf_files[0])
        self.assertGreater(len(doc.text), 0)
        self.assertGreater(doc.metadata["page_count"], 0)
        self.assertTrue(all(page.page_number is not None for page in doc.pages))


if __name__ == "__main__":
    unittest.main()

