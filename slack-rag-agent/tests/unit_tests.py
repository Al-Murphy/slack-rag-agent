import unittest

from processing.chunking import split_text
from processing.text_extractor import extract_text_from_html


class UnitTests(unittest.TestCase):
    def test_split_text_creates_chunks(self) -> None:
        text = "word " * 500
        chunks = split_text(text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(c, str) and c for c in chunks))

    def test_extract_text_from_html(self) -> None:
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        self.assertIn("Title", text)
        self.assertIn("Hello world", text)


if __name__ == "__main__":
    unittest.main()
