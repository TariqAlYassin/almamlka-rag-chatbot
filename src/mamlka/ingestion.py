from pathlib import Path
import re

import pymupdf


# Examples:
# 1. Introduction
# 2. Project Goals
# 3.1 Technical Requirements
# 4) Project Timeline
HEADING_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S.*$")


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each PDF page, preserving page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()

            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_number,
                        "source": pdf_path.name,
                    }
                )

    return pages


def split_into_sections(pages: list[dict]) -> list[dict]:
    """Split extracted text whenever a numbered heading is encountered."""
    sections = []

    current_heading = "Document Introduction"
    current_lines = []
    current_pages = set()

    def save_section():
        content = "\n".join(current_lines).strip()

        if not content:
            return

        sections.append(
            {
                "heading": current_heading,
                "text": f"{current_heading}\n{content}",
                "metadata": {
                    "pages": sorted(current_pages),
                },
            }
        )

    for page in pages:
        page_number = page["page"]

        for raw_line in page["text"].splitlines():
            line = raw_line.strip()

            if not line:
                continue

            if HEADING_PATTERN.match(line):
                save_section()

                current_heading = line
                current_lines = []
                current_pages = {page_number}
            else:
                current_lines.append(line)
                current_pages.add(page_number)

    save_section()
    return sections


def extract_sections(pdf_path: Path) -> list[dict]:
    """Extract heading-based sections from one PDF."""
    pages = extract_pages(pdf_path)
    sections = split_into_sections(pages)

    for section in sections:
        section["metadata"]["source"] = pdf_path.name

    return sections


def load_pdfs(pdf_directory: Path) -> list[dict]:
    """Load and section all PDFs in a directory."""
    if not pdf_directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_directory}")

    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    sections = []

    for pdf_path in pdf_files:
        sections.extend(extract_sections(pdf_path))

    return sections
