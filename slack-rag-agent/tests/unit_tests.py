import unittest

from agent.controller import Plan, confidence_score, rerank_chunks
from processing.chunking import is_low_information, split_text
from slack_listener.channel_crawler import extract_paper_urls, extract_pdf_file_ids, parse_channel_targets
from processing.text_extractor import extract_text_from_html


class FakeChunk:
    def __init__(self, content: str, section: str = "results") -> None:
        self.content = content
        self.section = section
        self.id = 1
        self.doc_id = "doc-1"


class UnitTests(unittest.TestCase):
    def test_split_text_creates_chunks(self) -> None:
        text = ("This is a results paragraph with meaningful scientific detail. " * 1200).strip()
        chunks = split_text(text, max_tokens=900, min_tokens=500, overlap_tokens=90)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(c, str) and len(c.split()) >= 500 for c in chunks[:-1]))

    def test_extract_text_from_html(self) -> None:
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        self.assertIn("Title", text)
        self.assertIn("Hello world", text)

    def test_low_information_filter(self) -> None:
        self.assertTrue(is_low_information("---- .... 123"))
        self.assertFalse(is_low_information("This paper reports strong effect sizes in the treatment cohort. " * 10))

    def test_reranker_and_confidence(self) -> None:
        chunks = [
            FakeChunk("The results show improved recall and precision in the benchmark.", section="results"),
            FakeChunk("Methods include randomized training and ablation studies.", section="methods"),
        ]
        ranked = rerank_chunks("What results improved precision?", chunks)
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])
        conf = confidence_score(Plan("evidence", 2, 0.4), ranked)
        self.assertGreater(conf, 0.0)

    def test_extract_pdf_file_ids(self) -> None:
        messages = [
            {"files": [{"id": "F1", "mimetype": "application/pdf"}]},
            {"files": [{"id": "F2", "filetype": "pdf"}]},
            {"files": [{"id": "F1", "mimetype": "application/pdf"}]},
            {"files": [{"id": "F3", "mimetype": "image/png"}]},
        ]
        self.assertEqual(extract_pdf_file_ids(messages), ["F1", "F2"])

    def test_parse_channel_targets(self) -> None:
        self.assertEqual(parse_channel_targets(["C1", " C2 "]), ["C1", "C2"])

    def test_extract_paper_urls(self) -> None:
        messages = [
            {"text": "Check this https://arxiv.org/abs/1706.03762"},
            {"text": "<https://doi.org/10.1038/s41586-020-2649-2|paper>"},
            {"text": "Ignore this https://example.com/home"},
        ]
        urls = extract_paper_urls(messages)
        self.assertEqual(len(urls), 2)


if __name__ == "__main__":
    unittest.main()
