from pathlib import Path

from mamlka.ingestion import HEADING_PATTERN, load_pdfs


def test_heading_detection():
    examples = [
        "1. Introduction",
        "2. Project Goals",
        "3.1 Technical Requirements",
        "4) Project Timeline",
    ]

    for heading in examples:
        assert HEADING_PATTERN.match(heading), f"Failed to detect heading: {heading}"


def test_pdf_ingestion():
    sections = load_pdfs(Path("data/pdfs"))

    assert sections, "No sections extracted"

    for section in sections:
        assert section["heading"]
        assert section["text"]
        assert section["metadata"]["source"]
        assert section["metadata"]["pages"]

    for section in sections[:5]:
        print(
            section["heading"],
            section["metadata"],
        )


if __name__ == "__main__":
    test_heading_detection()
    test_pdf_ingestion()
    print("All ingestion tests passed.")
