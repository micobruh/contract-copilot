import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType


PAGE_BREAK = "<<<PAGE_BREAK>>>"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

SECTION_HEADER_RE = re.compile(
    r"(?m)^(?:##\s*)?(\d{1,2})\.\s+(?!\d)([^\n]{1,200})$"
)

ENTITY_SUFFIXES = (
    r'Inc\.?|LLC|L\.L\.C\.|Corp\.?|Corporation|Company|Ltd\.?|Limited|'
    r'LP|L\.P\.|LLP|L\.L\.P\.|PLC|Bank|N\.A\.|National Association'
)

LEGAL_ENTITY_RE = re.compile(
    rf"""
    (?<![A-Za-z0-9&])
    (
        [A-Z][A-Za-z0-9&.,'\-()]* 
        (?:\s+[A-Z][A-Za-z0-9&.,'\-()]*){{0,8}}
        \s+(?:{ENTITY_SUFFIXES})
    )
    (?=[,)\s])
    """,
    re.VERBOSE,
)

BAD_SUBSTRINGS = [
    "dated as of",
    "effective date",
    "entered into by",
    "whereas",
    "has agreed to",
    "used exclusively",
    "and together with",
    "the company",
    "company subsidiaries",
    "buyer entities",
    "field",
    "agreement",
]


def infer_company_from_filename(file_path: str) -> str | None:
    stem = Path(file_path).stem

    # take first chunk before date-like pattern
    m = re.match(r'([A-Za-z0-9]+?)(?:_[0-9]{8}|$)', stem)
    if not m:
        return None

    raw = m.group(1)

    # split camel-ish company endings
    raw = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw)
    raw = re.sub(r'(Inc|Corp|Corporation|LLC|Ltd|PLC|Bank)$', r' \1', raw)
    raw = raw.replace('_', ' ').strip()

    return raw or None


def prettify_title(title: str | None) -> str | None:
    if not title:
        return None
    return title.title()


def clean_company_name(name: str) -> str:
    name = " ".join(name.split()).strip(" ,;.")

    # Remove OCR-ish aliases / quoted labels
    name = re.sub(r'\([^)]{0,60}\)$', '', name).strip(" ,;.")

    # Remove descriptive tails
    name = re.sub(
        r'\s+(and together with|together with|collectively|each of|on the one hand|on the other hand)\b.*$',
        '',
        name,
        flags=re.IGNORECASE
    ).strip(" ,;.")

    return name


def looks_like_real_company(name: str) -> bool:
    if not name:
        return False

    lowered = name.lower()

    if any(bad in lowered for bad in BAD_SUBSTRINGS):
        return False

    # Must end with a legal suffix
    if not re.search(r'\b(inc\.?|llc|corp\.?|corporation|company|ltd\.?|limited)\b$', lowered):
        return False

    # Too long usually means it swallowed prose
    if len(name.split()) > 8:
        return False

    return True


def dedupe_companies(companies: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in companies:
        key = re.sub(r'[^a-z0-9]+', '', c.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def get_intro_block(text: str) -> str:
    parts = re.split(r'(?im)^\s*WHEREAS[, ]', text, maxsplit=1)
    intro = parts[0]
    return intro[: 2500]


def extract_all_companies_from_intro(text: str) -> List[str]:
    intro = get_intro_block(text)
    matches = LEGAL_ENTITY_RE.findall(intro)

    companies = []
    seen = set()

    for m in matches:
        c = clean_company_name(m)
        key = re.sub(r'[^a-z0-9]+', '', c.lower())
        if c and key not in seen:
            seen.add(key)
            companies.append(c)

    return companies


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def decode_tokens(token_ids: List[int]) -> str:
    return TOKENIZER.decode(token_ids, skip_special_tokens=True).strip()


def normalize_false_markdown_subsection_headers(text: str) -> str:
    text = re.sub(
        r'(?m)^##\s*(\d+\.\d+(?:\.\d+)*)\s+',
        r'\1 ',
        text,
    )
    text = re.sub(
        r'(?m)^##\s*(\d+)\.\s+(\d+(?:\.\d+)*)\s+',
        r'\1.\2 ',
        text,
    )
    return text


def normalize_inline_subsection_prefixes(text: str) -> str:
    text = re.sub(
        r'(?m)^\s*[-•·]\s+(\d+\.\d+(?:\.\d+)*)\s+',
        r'\1 ',
        text,
    )
    text = re.sub(
        r'(?m)^##\s*(\d+\.\d+(?:\.\d+)*)\s+',
        r'\1 ',
        text,
    )
    text = re.sub(
        r'(?m)^###\s*(\d+\.\d+(?:\.\d+)*)\s+',
        r'\1 ',
        text,
    )
    return text


def normalize_broken_subsection_numbers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text

        # 13. 5 -> 13.5
        text = re.sub(
            r'(?m)^(\s*(?:#{2,3}\s*)?(?:[-•]\s*)?)(\d+)\.\s+(\d+)(?=\s)',
            r'\1\2.\3',
            text,
        )

        # 10. 2. 3 -> 10.2.3
        text = re.sub(
            r'(?m)^(\s*(?:#{2,3}\s*)?(?:[-•]\s*)?)(\d+\.\d+)\.\s+(\d+)(?=\s)',
            r'\1\2.\3',
            text,
        )

    return text


def promote_top_level_section_headers(text: str) -> str:
    return re.sub(
        r'(?m)^(?!(##\s))(\d{1,2}\.\s+(?!\d)[^\n]{1,200})$',
        r'## \2',
        text,
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u00a0": " ",
        ";": ";",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "·": "-",
        "•": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 1. Remove wrongly inserted markdown headings on subsections
    text = normalize_false_markdown_subsection_headers(text)

    # 2. Fix broken numbering like 13. 5 -> 13.5
    text = normalize_broken_subsection_numbers(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Fix malformed true headings like ## 5.Term
    text = re.sub(r"(?m)^(##\s*)(\d+)\.(\S)", r"\1\2. \3", text)

    # 4. Promote only true top-level sections
    text = re.sub(
        r'(?m)^(?!(##\s))(\d{1,2}\.\s+(?!\d)[^\n]{1,200})$',
        r'## \2',
        text,
    )

    text = re.sub(r"(?m)^Appendix\s*$", "## Appendix", text)

    text = normalize_inline_subsection_prefixes(text)

    return text.strip()


def detect_metadata_from_pages(pages: List[Tuple[int, str]]) -> Dict[str, str]:
    text = "\n\n".join(page_text for _, page_text in pages)
    meta: Dict[str, str] = {}

    m = re.search(r"(?m)^##\s+([A-Z][A-Z\s&\-]+)$", text)
    if m:
        meta["document_title"] = m.group(1).strip()

    return meta


def is_appendix_heading(line: str) -> bool:
    return bool(re.match(r"(?i)^##\s*appendix\s*$", line.strip()))


def split_into_section_blocks(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    current = {
        "section_number": None,
        "section_title": "Front Matter",
        "section_type": "front_matter",
        "content_parts": [],
        "page_numbers": [],
    }

    in_appendix = False

    for page_no, page_text in pages:
        lines = page_text.splitlines()

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                if current["content_parts"] and current["content_parts"][-1] != "":
                    current["content_parts"].append("")
                continue

            if is_appendix_heading(line):
                if current["content_parts"]:
                    sections.append(
                        {
                            "section_number": current["section_number"],
                            "section_title": current["section_title"],
                            "section_type": current["section_type"],
                            "text": "\n".join(current["content_parts"]).strip(),
                            "page_numbers": sorted(set(current["page_numbers"])),
                        }
                    )

                in_appendix = True
                current = {
                    "section_number": None,
                    "section_title": "Appendix",
                    "section_type": "appendix",
                    "content_parts": ["## Appendix"],
                    "page_numbers": [page_no],
                }
                continue

            m = SECTION_HEADER_RE.match(line)
            if m and not in_appendix:
                if current["content_parts"]:
                    sections.append(
                        {
                            "section_number": current["section_number"],
                            "section_title": current["section_title"],
                            "section_type": current["section_type"],
                            "text": "\n".join(current["content_parts"]).strip(),
                            "page_numbers": sorted(set(current["page_numbers"])),
                        }
                    )

                sec_num = m.group(1).strip()
                sec_title = m.group(2).strip()

                current = {
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "section_type": "body",
                    "content_parts": [f"## {sec_num}. {sec_title}"],
                    "page_numbers": [page_no],
                }
            else:
                current["content_parts"].append(line)
                current["page_numbers"].append(page_no)

    if current["content_parts"]:
        sections.append(
            {
                "section_number": current["section_number"],
                "section_title": current["section_title"],
                "section_type": current["section_type"],
                "text": "\n".join(current["content_parts"]).strip(),
                "page_numbers": sorted(set(current["page_numbers"])),
            }
        )

    return [s for s in sections if s["text"].strip()]


def split_pages(markdown_text: str) -> List[Tuple[int, str]]:
    pages = markdown_text.split(PAGE_BREAK)
    out = []
    for i, page in enumerate(pages, start=1):
        page = normalize_text(page)
        if page.strip():
            out.append((i, page.strip()))
    return out


def make_subsection_regex_for_section(section_number: str) -> re.Pattern:
    escaped = re.escape(section_number)
    return re.compile(
        rf'(?:(?<=^)|(?<=\n)|(?<=\n\n))'
        rf'(?:[-•]\s*|#{{2,3}}\s*)?'
        rf'({escaped}\.\d+(?:\.\d+)*)'
        rf'(?=\s)',
        re.MULTILINE
    )


def is_heading_only_prefix(prefix: str, section_number: str, section_title: str) -> bool:
    normalized_prefix = " ".join(prefix.split()).strip()
    heading = f"## {section_number}. {section_title}"
    normalized_heading = " ".join(heading.split()).strip()

    return normalized_prefix == normalized_heading


def split_section_into_inline_subsections(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    if section["section_type"] != "body" or not section.get("section_number"):
        return [dict(section, subsection_number=None, subsection_title=None)]

    text = section["text"]
    subsection_re = make_subsection_regex_for_section(section["section_number"])
    matches = list(subsection_re.finditer(text))

    if not matches:
        return [dict(section, subsection_number=None, subsection_title=None)]

    units = []
    first_start = matches[0].start()
    prefix = text[:first_start].strip()

    heading = f"## {section['section_number']}. {section['section_title']}".strip()
    if prefix and prefix != heading:
        units.append({
            "section_number": section["section_number"],
            "section_title": section["section_title"],
            "subsection_number": None,
            "subsection_title": None,
            "section_type": section["section_type"],
            "text": prefix,
            "page_numbers": section["page_numbers"],
        })

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        block = re.sub(r'^(?:[-•]\s*|#{2,3}\s*)', '', block).strip()

        units.append({
            "section_number": section["section_number"],
            "section_title": section["section_title"],
            "subsection_number": match.group(1).strip(),
            "subsection_title": None,
            "section_type": section["section_type"],
            "text": block,
            "page_numbers": section["page_numbers"],
        })

    return units

def split_appendix_into_blocks(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = section["text"]
    lines = text.splitlines()

    blocks = []
    current_title = "Appendix"
    current_lines = []
    page_numbers = section["page_numbers"]

    def flush():
        if current_lines:
            blocks.append(
                {
                    "section_number": None,
                    "section_title": current_title,
                    "subsection_number": None,
                    "subsection_title": None,
                    "section_type": "appendix",
                    "text": "\n".join(current_lines).strip(),
                    "page_numbers": page_numbers,
                }
            )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        looks_like_subheading = (
            not stripped.startswith("-")
            and len(stripped) < 120
            and (
                "Restricted Key Words" in stripped
                or "Restricted Trademark Terms" in stripped
                or re.match(
                    r"^[A-Z][A-Za-z0-9/&'\"(). -]{2,40}(?:\s+[A-Z][A-Za-z0-9/&'\"(). -]{2,40}){0,4}$",
                    stripped,
                )
            )
        )

        if looks_like_subheading and current_lines:
            flush()
            current_title = stripped
            current_lines = [f"### {stripped}"]
        else:
            current_lines.append(stripped)

    flush()

    if not blocks:
        return [dict(section, subsection_number=None, subsection_title=None)]

    return blocks


def build_semantic_units(
    markdown_text: str,
    source_file: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    pages = split_pages(markdown_text)
    base_metadata = detect_metadata_from_pages(pages)
    sections = split_into_section_blocks(pages)

    semantic_units: List[Dict[str, Any]] = []

    for section in sections:
        if section["section_type"] == "appendix":
            units = split_appendix_into_blocks(section)
        else:
            units = split_section_into_inline_subsections(section)

        for unit in units:
            semantic_unit_id_parts = [
                Path(source_file).stem,
                unit.get("section_number") or "NA",
                unit.get("subsection_number") or "NA",
                unit.get("section_type") or "NA",
            ]
            semantic_unit_id = "|".join(semantic_unit_id_parts)

            heading_lines = []
            if unit.get("section_number") and unit.get("section_title"):
                heading_lines.append(f"## {unit['section_number']}. {unit['section_title']}")
            elif unit.get("section_type") == "appendix":
                heading_lines.append(f"## {unit['section_title']}")
            elif unit.get("section_type") == "front_matter":
                heading_lines.append("## Front Matter")

            if unit.get("subsection_number") and unit.get("subsection_title"):
                heading_lines.append(f"### {unit['subsection_number']} {unit['subsection_title']}")

            body_text = unit["text"].strip()

            # Avoid duplicating heading if already present at start
            full_text = body_text
            heading_prefix = "\n".join(heading_lines).strip()
            if heading_prefix and not body_text.startswith(heading_prefix):
                full_text = f"{heading_prefix}\n\n{body_text}".strip()

            semantic_units.append(
                {
                    **unit,
                    "semantic_unit_id": semantic_unit_id,
                    "text": full_text,
                    "token_count": count_tokens(full_text),
                    "source": source_file,
                }
            )

    return semantic_units, base_metadata


def split_paragraph_by_tokens(
    paragraph: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    token_ids = TOKENIZER.encode(paragraph, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [paragraph.strip()]

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_text = decode_tokens(token_ids[start:end])
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(token_ids):
            break
        start = max(0, end - overlap_tokens)

    return chunks


def split_semantic_unit_to_child_chunks(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
) -> List[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return split_paragraph_by_tokens(text, max_tokens, overlap_tokens)

    child_chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            if current_parts:
                child_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_tokens = 0

            child_chunks.extend(
                split_paragraph_by_tokens(
                    para,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue

        if current_tokens + para_tokens <= max_tokens:
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            if current_parts:
                child_chunks.append("\n\n".join(current_parts).strip())

            current_parts = [para]
            current_tokens = para_tokens

    if current_parts:
        child_chunks.append("\n\n".join(current_parts).strip())

    return child_chunks


def build_final_documents(
    semantic_units: List[Dict[str, Any]],
    base_metadata: Dict[str, str],
    max_tokens: int = 256,
    overlap_tokens: int = 40,
) -> List[Document]:
    docs: List[Document] = []

    for unit in semantic_units:
        child_texts = split_semantic_unit_to_child_chunks(
            text=unit["text"],
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        child_ids = [
            f"{unit['semantic_unit_id']}|chunk|{i+1}"
            for i in range(len(child_texts))
        ]

        for i, child_text in enumerate(child_texts):
            chunk_id = child_ids[i]
            prev_chunk_id = child_ids[i - 1] if i > 0 else None
            next_chunk_id = child_ids[i + 1] if i < len(child_ids) - 1 else None

            metadata = {
                "chunk_id": chunk_id,
                "semantic_unit_id": unit["semantic_unit_id"],
                "source": unit["source"],
                "document_title": base_metadata.get("document_title"),
                "all_companies": base_metadata.get("all_companies"),
                "party_count": base_metadata.get("party_count"),
                "section_number": unit.get("section_number"),
                "section_title": unit.get("section_title"),
                "subsection_number": unit.get("subsection_number"),
                "subsection_title": unit.get("subsection_title"),
                "section_type": unit.get("section_type"),
                "page_numbers": unit.get("page_numbers"),
                "semantic_unit_token_count": unit.get("token_count"),
                "token_count": count_tokens(child_text),
                "child_chunk_index": i + 1,
                "child_chunk_count": len(child_texts),
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
            }

            docs.append(Document(page_content=child_text, metadata=metadata))

    return docs


def get_pdf_path(relative_path: str) -> Path:
    root_path = Path(__file__).resolve().parents[3]
    return root_path / Path(relative_path)


def load_pdf(
    relative_path: str,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
) -> List[Document]:
    pdf_path = get_pdf_path(relative_path)
    assert pdf_path.exists(), f"File not found: {pdf_path}"

    loader = DoclingLoader(
        file_path=str(pdf_path),
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={
            "page_break_placeholder": f"\n\n{PAGE_BREAK}\n\n",
        },
    )

    raw_docs = loader.load()

    if not raw_docs:
        return []

    markdown_text = raw_docs[0].page_content

    semantic_units, base_metadata = build_semantic_units(
        markdown_text=markdown_text,
        source_file=str(relative_path),
    )

    for unit in semantic_units:
        if unit["section_type"] in {"front_matter", "body"} and unit["text"].strip():
            first_text = unit["text"][: 2000]
            break

    companies = extract_all_companies_from_intro(first_text)
    company_info = {
        "document_title": base_metadata.get("document_title"),
        "all_companies": companies,
        "party_count": len(companies),
    }       

    base_metadata.update(company_info)    

    print(f"Built {len(semantic_units)} semantic units.")
    for i, unit in enumerate(semantic_units[: 2], start=1):
        print("-" * 80)
        print(f"Unit {i}")
        print(f"Document Title: {base_metadata.get('document_title')}")
        print(f"All Companies: {base_metadata.get('all_companies')}")
        print(f"Party Count: {base_metadata.get('party_count')}")
        print(f"Source: {unit['source']}")
        print(f"Semantic Unit ID: {unit['semantic_unit_id']})")
        print(f"Section: {unit.get('section_number')} - {unit.get('section_title')}")
        print(f"Token Count: {unit['token_count']}")
        # print(unit["text"])

    docs = build_final_documents(
        semantic_units=semantic_units,
        base_metadata=base_metadata,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    # print(f"Built {len(docs)} final chunks.")
    # for i, doc in enumerate(docs, start=1):
    #     print("-" * 80)
    #     print(f"Chunk {i}")
    #     print(doc.metadata)
    #     print(doc.page_content)

    return docs

# relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
# relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf"
relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf"
# relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/IP/ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual Property Agreement.pdf"
load_pdf(relative_path)
# python src/contract_copilot/indexer/ocr_loader.py