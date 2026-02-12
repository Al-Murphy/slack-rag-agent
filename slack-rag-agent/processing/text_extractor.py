from bs4 import BeautifulSoup


def extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
