import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType


PAGE_BREAK = "<<<PAGE_BREAK>>>"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_DIR = Path("data/raw/CUAD_v1/full_contract_pdf")


# ---------- Regexes ----------

# Previous format
TOP_SECTION_RE = re.compile(
    r"(?m)^(?:##\s*)?(\d{1,2})\.\s+(?!\d)([^\n]{1,200})$"
)

# New format
ARTICLE_RE = re.compile(r"(?m)^ARTICLE\s+([IVXLC]+)\s*$", re.IGNORECASE)
SECTION_XY_RE = re.compile(
    r"(?im)^Section\s+(\d+\.\d+)\s+(.{1,140}?)(?:\.\s*$|\s*$)"
)

ROMAN_SECTION_RE = re.compile(
    r"(?im)^##\s*([IVXLC]+)\.\s+(.{1,120}?)\s*$"
)
PARAGRAPH_SECTION_RE = re.compile(
    r"(?im)^(?:##\s*)?§\s*(\d+(?:\.\d+)*)\s+(.{1,160}?)(?=\s*\(\d+\)\s+|\.\s*$|$)"
)
LETTER_SECTION_RE = re.compile(
    r"(?im)^(?:##\s*)?([A-Z])\.\s+(.{1,160})$"
)

# Optional secondary splitters inside long sections
LETTER_SUBUNIT_RE = re.compile(
    r"(?im)^\(([a-z])\)\s+([^.\n]{1,100})(?:\.)?(?:\s|$)"
)
ROMAN_SUBUNIT_RE = re.compile(
    r"(?im)^\(((?:ix|iv|v?i{1,3}|x))\)\s+([^.\n:;]{1,120})(?:[.:;])?(?:\s|$)"
)
NUMBERED_ITEM_START_RE = re.compile(
    r"(?im)^\((\d+)\)\s+(.{1,120}?)(?=\.\s|:\s|$)"
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

APPENDIX_HEADING_RE = re.compile(r"(?im)^##\s*Appendix\s*$|^Appendix\s*$")
APPENDIX_SUBHEAD_RE = re.compile(
    r"(?im)^##\s+(.+)$|^([A-Z][A-Za-z0-9/&'\"(). -]{2,120})$"
)


# ---------- Company finder ----------

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


# ---------- Token helpers ----------

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def decode_tokens(token_ids: List[int]) -> str:
    return get_tokenizer().decode(token_ids, skip_special_tokens=True).strip()


# ---------- Normalization ----------

def strip_parser_artifacts(text: str) -> str:
    # Remove parser/source lines
    text = re.sub(
        r'(?m)^Source:\s+.*?<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>\s*',
        '',
        text,
    )

    # Remove standalone page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    return text


def promote_paragraph_section_headings(text: str) -> str:
    return re.sub(
        r'(?im)^(?!##\s)(§\s*\d+(?:\.\d+)*\s+.{1,160}?)(?:\s*(?=\(\d+\))|\s*$)',
        r'## \1',
        text,
    )


def build_parent_heading_prefix(unit: Dict[str, Any]) -> str:
    parts = []

    if unit.get("roman_section_number") and unit.get("roman_section_title"):
        parts.append(f"## {unit['roman_section_number']}. {unit['roman_section_title']}")

    if unit.get("section_number") and unit.get("section_title"):
        parts.append(f"## § {unit['section_number']} {unit['section_title']}")

    return "\n".join(parts).strip()


def inject_heading_breaks(text: str) -> str:
    # ARTICLE headings
    text = re.sub(
        r'(?<!\n)(ARTICLE\s+[IVXLC]+\b)',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )

    # Section X.Y headings
    text = re.sub(
        r'(?<!\n)(Section\s+\d+\.\d+\s+)',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )

    # Roman markdown headings
    text = re.sub(
        r'(?<!\n)(##\s*[IVXLC]+\.\s+[A-Z])',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )

    # Only markdown-prefixed § headings
    text = re.sub(
        r'(?<!\n)(##\s*§\s*\d+(?:\.\d+)*\s+[A-Z])',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )

    # Optional numbered list items only at obvious boundaries
    text = re.sub(r'(?<!\n)(\(\d+\)\s+)', r'\n\1', text)

    # Then promote bare § headings only if already at line start
    text = promote_paragraph_section_headings(text)

    return text


def inject_paragraph_section_breaks(text: str) -> str:
    # Insert a newline before bare § headings only when they look like real headings,
    # not inline references like "(see above § 3)".
    text = re.sub(
        r'(?<!\n)(§\s*\d+(?:\.\d+)*\s+[A-Z][^\n]{1,120}?)(?=\s+\(\d+\)|\.\s|$)',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )
    return text

def normalize_missing_space_after_number_period(text: str) -> str:
    """
    Fix cases like:
      ## 5.Term of this Agreement   -> ## 5. Term of this Agreement
      13.5Governing Law             -> 13.5 Governing Law   (optional)
      § 4Independence               -> § 4 Independence     (optional)
    """

    # Top-level numbered headings: 5.Term -> 5. Term
    text = re.sub(
        r'(?m)^(\s*##\s*\d+)\.([A-Za-z])',
        r'\1. \2',
        text,
    )

    # Bare top-level headings without ##
    text = re.sub(
        r'(?m)^(\s*\d+)\.([A-Za-z])',
        r'\1. \2',
        text,
    )

    # Optional: subsection headings like 13.5Governing -> 13.5 Governing
    text = re.sub(
        r'(?m)^(\s*(?:##\s*)?\d+\.\d+(?:\.\d+)*)\s*([A-Za-z])',
        r'\1 \2',
        text,
    )

    # Optional: § headings like § 4Independence -> § 4 Independence
    text = re.sub(
        r'(?m)^(\s*(?:##\s*)?§\s*\d+(?:\.\d+)*)\s*([A-Za-z])',
        r'\1 \2',
        text,
    )

    return text


def normalize_false_markdown_subsection_headers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(
            r'(?m)^##\s*(\d+)\.\s+(\d+(?:\.\d+)*)\s+',
            r'\1.\2 ',
            text,
        )
        text = re.sub(
            r'(?m)^##\s*(\d+\.\d+(?:\.\d+)*)\s+',
            r'\1 ',
            text,
        )
    return text


def normalize_broken_subsection_numbers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(
            r'(?m)^(\s*(?:#{2,3}\s*)?(?:[-•]\s*)?)(\d+)\.\s+(\d+)(?=\s)',
            r'\1\2.\3',
            text,
        )
        text = re.sub(
            r'(?m)^(\s*(?:#{2,3}\s*)?(?:[-•]\s*)?)(\d+\.\d+)\.\s+(\d+)(?=\s)',
            r'\1\2.\3',
            text,
        )
    return text


def normalize_inline_subsection_prefixes(text: str) -> str:
    text = re.sub(r'(?m)^\s*[-•·]\s+(\d+\.\d+(?:\.\d+)*)\s+', r'\1 ', text)
    text = re.sub(r'(?m)^##\s*(\d+\.\d+(?:\.\d+)*)\s+', r'\1 ', text)
    text = re.sub(r'(?m)^###\s*(\d+\.\d+(?:\.\d+)*)\s+', r'\1 ', text)
    return text


def normalize_section_sign_headings(text: str) -> str:
    """
    Fix OCR/markdown issues around § headings.

    Examples:
    ## § 9 Final provisions(1) Force majeure
    -> ## § 9 Final provisions
       (1) Force majeure

    ##§ 8 Foo
    -> ## § 8 Foo
    """
    # Normalize spacing after ##
    text = re.sub(r'(?im)^##\s*§', '## §', text)

    # Put newline before markdown § heading if glued after previous text
    text = re.sub(r'(?<!\n)(##\s*§\s*\d+(?:\.\d+)*)', r'\n\1', text, flags=re.IGNORECASE)

    # Put newline between § heading title and first numbered item if glued
    text = re.sub(
        r'(?im)^(##\s*§\s*\d+(?:\.\d+)*\s+.*?)(\(\d+\)\s+)',
        r'\1\n\2',
        text,
    )

    # Also support bare § headings
    text = re.sub(
        r'(?im)^(§\s*\d+(?:\.\d+)*\s+.*?)(\(\d+\)\s+)',
        r'\1\n\2',
        text,
    )

    return text


def inject_numbered_item_breaks(text: str) -> str:
    """
    Ensure numbered items start on a new line when OCR glued them inline.
    """
    # If a numbered item appears after prose, split it to a new line
    text = re.sub(r'(?<!\n)\s(\((?:\d+)\)\s+)', r'\n\1', text)

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

    text = normalize_missing_space_after_number_period(text)

    text = strip_parser_artifacts(text)
    text = inject_heading_breaks(text)
    text = normalize_section_sign_headings(text)
    text = inject_numbered_item_breaks(text)

    text = normalize_false_markdown_subsection_headers(text)
    text = normalize_broken_subsection_numbers(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\(\s*([a-zivx\d]+)\s*\)", r"(\1)", text, flags=re.IGNORECASE)

    text = re.sub(r"(?m)^Appendix\s*$", "## Appendix", text)

    return text.strip()


def split_pages(markdown_text: str) -> List[Tuple[int, str]]:
    pages = markdown_text.split(PAGE_BREAK)
    out = []
    for i, page in enumerate(pages, start=1):
        page = normalize_text(page)
        if page.strip():
            out.append((i, page.strip()))
    return out


# ---------- Metadata ----------

def detect_metadata_from_pages(pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    text = "\n\n".join(page for _, page in pages)
    meta: Dict[str, Any] = {}
    m = re.search(r"(?m)^##\s+([A-Z][A-Z\s&\-]+)$", text)
    if m:
        meta["document_title"] = m.group(1).strip()

    return meta


def extract_path_metadata(source_file: str) -> Dict[str, Any]:
    path = Path(source_file)
    parts = path.parts

    dataset = next((part for part in parts if part.endswith("_v1")), None)
    try:
        full_contract_index = parts.index("full_contract_pdf")
    except ValueError:
        full_contract_index = -1

    part = parts[full_contract_index + 1] if full_contract_index >= 0 and len(parts) > full_contract_index + 1 else None
    contract_type = parts[full_contract_index + 2] if full_contract_index >= 0 and len(parts) > full_contract_index + 2 else None

    document_id = path.stem
    return {
        "document_id": document_id,
        "source_path": source_file,
        "file_name": path.name,
        "dataset": dataset,
        "corpus": "full_contract_pdf" if full_contract_index >= 0 else None,
        "part": part,
        "contract_type": contract_type,
    }


def build_document_metadata(
    base_metadata: Dict[str, Any],
    source_file: str,
    semantic_units: List[Dict[str, Any]],
) -> Dict[str, Any]:
    source_metadata = extract_path_metadata(source_file)

    first_text = next(
        (
            unit["text"][:2000]
            for unit in semantic_units
            if unit.get("section_type") in {"front_matter", "body"} and unit.get("text", "").strip()
        ),
        "",
    )

    companies = dedupe_companies(extract_all_companies_from_intro(first_text))
    if not companies:
        inferred_company = infer_company_from_filename(source_file)
        if inferred_company:
            companies = [inferred_company]

    document_title = base_metadata.get("document_title") or prettify_title(source_metadata["document_id"])

    return {
        **base_metadata,
        **source_metadata,
        "document_title": document_title,
        "title": document_title,
        "company_names": companies,
        "all_companies": companies,
        "party_count": len(companies),
    }


# ---------- Format detection ----------

def count_article_section_markers(text: str) -> int:
    return len(ARTICLE_RE.findall(text)) + len(SECTION_XY_RE.findall(text))


def count_numeric_section_markers(text: str) -> int:
    return len(TOP_SECTION_RE.findall(text))


# ---------- Parser 1: numeric section format ----------

def make_subsection_regex_for_section(section_number: str) -> re.Pattern:
    escaped = re.escape(section_number)
    return re.compile(
        rf'(?:(?<=^)|(?<=\n)|(?<=\n\n))'
        rf'(?:[-•]\s*|#{{2,3}}\s*)?'
        rf'({escaped}\.\d+(?:\.\d+)*)'
        rf'(?=\s)',
        re.MULTILINE
    )


def is_appendix_heading(line: str) -> bool:
    return bool(APPENDIX_HEADING_RE.match(line.strip()))


def split_numeric_top_sections(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
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
        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                if current["content_parts"] and current["content_parts"][-1] != "":
                    current["content_parts"].append("")
                continue

            # Hard boundary: Appendix starts
            if is_appendix_heading(line):
                if current["content_parts"]:
                    sections.append({
                        **current,
                        "text": "\n".join(current["content_parts"]).strip(),
                        "page_numbers": sorted(set(current["page_numbers"])),
                    })

                in_appendix = True
                current = {
                    "section_number": None,
                    "section_title": "Appendix",
                    "section_type": "appendix",
                    "content_parts": ["## Appendix"],
                    "page_numbers": [page_no],
                }
                continue

            # Only detect numbered top-level sections before appendix
            if not in_appendix:
                m = TOP_SECTION_RE.match(line)
                if m:
                    if current["content_parts"]:
                        sections.append({
                            **current,
                            "text": "\n".join(current["content_parts"]).strip(),
                            "page_numbers": sorted(set(current["page_numbers"])),
                        })

                    current = {
                        "section_number": m.group(1).strip(),
                        "section_title": m.group(2).strip(),
                        "section_type": "body",
                        "content_parts": [f"## {m.group(1).strip()}. {m.group(2).strip()}"],
                        "page_numbers": [page_no],
                    }
                    continue

            current["content_parts"].append(line)
            current["page_numbers"].append(page_no)

    if current["content_parts"]:
        sections.append({
            **current,
            "text": "\n".join(current["content_parts"]).strip(),
            "page_numbers": sorted(set(current["page_numbers"])),
        })

    return [s for s in sections if s["text"].strip()]


def infer_subsection_title_from_text(
    text: str,
    max_chars: int = 120,
    max_words: int = 14,
) -> Optional[str]:
    """
    Infer a subsection title only when the text after the subsection number
    contains a short substring terminated by '.' or ':'.

    Examples accepted:
      13.5 Governing Law; Jurisdiction; Waiver of Jury Trial.
      10.2 Indemnification:
    
    Examples rejected:
      12.1 MA recognizes that the Technology in source form ...
    """
    text = " ".join(text.split()).strip()

    # Remove leading subsection number like 13.5 or 10.2.1
    text = re.sub(r"^\d+\.\d+(?:\.\d+)*\s+", "", text).strip()
    if not text:
        return None

    # Only accept a candidate if it ends at the first '.' or ':'
    m = re.match(r"(.+?)([.:])\s", text)
    if not m:
        # Also allow punctuation at end of string
        m = re.match(r"(.+?)([.:])$", text)
        if not m:
            return None

    candidate = m.group(1).strip(" ;,.-:")
    if not candidate:
        return None

    if len(candidate) > max_chars:
        return None

    if len(candidate.split()) > max_words:
        return None

    return candidate


def split_numeric_section_into_subsections(section: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            **section,
            "subsection_number": None,
            "subsection_title": None,
            "text": prefix,
        })

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        block = re.sub(r'^(?:[-•]\s*|#{2,3}\s*)', '', block).strip()

        subsection_number = match.group(1).strip()
        subsection_title = infer_subsection_title_from_text(block)

        units.append({
            **section,
            "subsection_number": subsection_number,
            "subsection_title": subsection_title,
            "text": block,
        })

    return units


# ---------- Parser 2: ARTICLE / Section X.Y format ----------

def strip_parent_heading_from_prefix(prefix: str, unit: Dict[str, Any]) -> str:
    text = prefix.strip()

    candidates = []

    if unit.get("roman_section_number") and unit.get("roman_section_title"):
        candidates.append(f"## {unit['roman_section_number']}. {unit['roman_section_title']}".strip())

    if unit.get("section_number") and unit.get("section_title"):
        candidates.append(f"## § {unit['section_number']} {unit['section_title']}".strip())
        candidates.append(f"§ {unit['section_number']} {unit['section_title']}".strip())
        candidates.append(f"Section {unit['section_number']} {unit['section_title']}".strip())
        candidates.append(f"## {unit['section_number']}. {unit['section_title']}".strip())

    for heading in candidates:
        if text.startswith(heading):
            return text[len(heading): ].strip()

    return text


def split_by_marker(
    unit: Dict[str, Any],
    marker_re: re.Pattern,
    label_key: str,
    title_key: str,
) -> List[Dict[str, Any]]:
    text = unit["text"]
    matches = list(marker_re.finditer(text))

    if not matches:
        return [dict(unit)]

    parts = []

    first_start = matches[0].start()
    prefix = text[: first_start].strip()

    if prefix:
        remainder = strip_parent_heading_from_prefix(prefix, unit)
        if remainder:
            parts.append({
                **unit,
                label_key: None,
                title_key: None,
                "text": prefix,
            })

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        parts.append({
            **unit,
            label_key: match.group(1).lower(),
            title_key: clean_subunit_title(match.group(2).strip()),
            "text": block,
        })

    return parts


def clean_heading_title(title: str, max_words: int = 14) -> str:
    title = " ".join(title.split()).strip(" .;,:")
    words = title.split()
    return " ".join(words[:max_words])


def clean_subunit_title(title: str, max_words: int = 10) -> str:
    title = " ".join(title.split()).strip(" .;,:")
    words = title.split()
    return " ".join(words[: max_words])


def split_article_section_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []

    current_article_number = None
    current_article_title = None

    current = {
        "section_type": "front_matter",
        "article_number": None,
        "article_title": None,
        "section_number": None,
        "section_title": "Front Matter",
        "content_parts": [],
        "page_numbers": [],
    }

    pending_article_title = False

    for page_no, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                if current["content_parts"] and current["content_parts"][-1] != "":
                    current["content_parts"].append("")
                continue

            m_article = ARTICLE_RE.match(line)
            if m_article:
                # close current body section before starting new article context
                if current["content_parts"] and current["section_type"] == "body":
                    units.append({
                        **current,
                        "text": "\n".join(current["content_parts"]).strip(),
                        "page_numbers": sorted(set(current["page_numbers"])),
                    })
                    current = {
                        "section_type": "front_matter",
                        "article_number": None,
                        "article_title": None,
                        "section_number": None,
                        "section_title": "Article Boundary",
                        "content_parts": [],
                        "page_numbers": [],
                    }

                current_article_number = m_article.group(1).upper()
                current_article_title = None
                pending_article_title = True
                continue

            if pending_article_title and line.isupper() and len(line) <= 120:
                current_article_title = line
                pending_article_title = False

                if current["section_number"] is None and current["section_type"] == "front_matter":
                    current["content_parts"].append(line)
                    current["page_numbers"].append(page_no)
                continue
            else:
                pending_article_title = False

            m_section = SECTION_XY_RE.match(line)
            if m_section:
                if current["content_parts"]:
                    units.append({
                        **current,
                        "text": "\n".join(current["content_parts"]).strip(),
                        "page_numbers": sorted(set(current["page_numbers"])),
                    })

                sec_num = m_section.group(1).strip()
                sec_title = clean_heading_title(m_section.group(2).strip())

                header_parts = []
                if current_article_number:
                    header_parts.append(f"ARTICLE {current_article_number}")
                if current_article_title:
                    header_parts.append(current_article_title)
                header_parts.append(f"Section {sec_num} {sec_title}")

                current = {
                    "section_type": "body",
                    "article_number": current_article_number,
                    "article_title": current_article_title,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "content_parts": header_parts,
                    "page_numbers": [page_no],
                }
            else:
                current["content_parts"].append(line)
                current["page_numbers"].append(page_no)

    if current["content_parts"]:
        units.append({
            **current,
            "text": "\n".join(current["content_parts"]).strip(),
            "page_numbers": sorted(set(current["page_numbers"])),
        })

    return [u for u in units if u["text"].strip()]


def split_article_section_into_subunits(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    if unit["section_type"] != "body":
        return [dict(unit, subunit_label=None, subunit_title=None, roman_subunit_label=None, roman_subunit_title=None)]

    # First split by (a), (b), (c)
    letter_units = split_by_marker(
        unit,
        LETTER_SUBUNIT_RE,
        "subunit_label",
        "subunit_title",
    )

    final_units = []
    for lu in letter_units:
        # Then split each letter unit by (i), (ii), (iii), (iv)
        roman_units = split_by_marker(
            lu,
            ROMAN_SUBUNIT_RE,
            "roman_subunit_label",
            "roman_subunit_title",
        )
        final_units.extend(roman_units)

    return final_units


def split_article_section_into_subunits(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    if unit["section_type"] != "body":
        return [dict(unit, subunit_label=None, subunit_title=None)]

    text = unit["text"]
    matches = list(LETTER_SUBUNIT_RE.finditer(text))

    if not matches:
        return [dict(unit, subunit_label=None, subunit_title=None)]

    parts = []

    first_start = matches[0].start()
    prefix = text[:first_start].strip()

    section_header = f"Section {unit['section_number']} {unit['section_title']}".strip()
    if prefix and section_header not in prefix:
        parts.append({
            **unit,
            "subunit_label": None,
            "subunit_title": None,
            "text": prefix,
        })

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start: end].strip()

        parts.append({
            **unit,
            "subunit_label": match.group(1).lower(),
            "subunit_title": match.group(2).strip(),
            "text": block,
        })

    return parts


def is_heading_only_unit(unit: Dict[str, Any]) -> bool:
    text = unit["text"].strip()
    if unit.get("roman_section_number") and not unit.get("section_number"):
        heading = f"## {unit['roman_section_number']}. {unit['roman_section_title']}".strip()
        return text == heading
    return False


def split_roman_paragraph_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []

    current_roman_number = None
    current_roman_title = None

    current = {
        "section_type": "front_matter",
        "roman_section_number": None,
        "roman_section_title": None,
        "section_number": None,
        "section_title": "Front Matter",
        "content_parts": [],
        "page_numbers": [],
    }

    def flush_current():
        nonlocal current
        if not current["content_parts"]:
            return

        candidate = {
            **current,
            "text": "\n".join(current["content_parts"]).strip(),
            "page_numbers": sorted(set(current["page_numbers"])),
        }

        if candidate["text"] and not is_heading_only_unit(candidate):
            units.append(candidate)

    for page_no, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                if current["content_parts"] and current["content_parts"][-1] != "":
                    current["content_parts"].append("")
                continue

            m_roman = ROMAN_SECTION_RE.match(line)
            if m_roman:
                flush_current()

                current_roman_number = m_roman.group(1).upper()
                current_roman_title = clean_heading_title(m_roman.group(2).strip())

                current = {
                    "section_type": "body",
                    "roman_section_number": current_roman_number,
                    "roman_section_title": current_roman_title,
                    "section_number": None,
                    "section_title": current_roman_title,
                    "content_parts": [f"## {current_roman_number}. {current_roman_title}"],
                    "page_numbers": [page_no],
                }
                continue

            m_letter = LETTER_SECTION_RE.match(line)
            if m_letter:
                flush_current()

                sec_letter = m_letter.group(1).strip()
                sec_title = clean_heading_title(m_letter.group(2).strip())

                current = {
                    "section_type": "body",
                    "roman_section_number": current_roman_number,
                    "roman_section_title": current_roman_title,
                    "section_number": sec_letter,   # <-- important
                    "section_title": sec_title,
                    "content_parts": [f"## {sec_letter}. {sec_title}"],
                    "page_numbers": [page_no],
                }
                continue

            m_para = PARAGRAPH_SECTION_RE.match(line)
            if m_para:
                flush_current()

                sec_num = m_para.group(1).strip()
                sec_title = clean_heading_title(m_para.group(2).strip())

                header_parts = []
                if current_roman_number and current_roman_title:
                    header_parts.append(f"## {current_roman_number}. {current_roman_title}")
                header_parts.append(f"## § {sec_num} {sec_title}")

                current = {
                    "section_type": "body",
                    "roman_section_number": current_roman_number,
                    "roman_section_title": current_roman_title,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "content_parts": header_parts,
                    "page_numbers": [page_no],
                }
                continue

            current["content_parts"].append(line)
            current["page_numbers"].append(page_no)

    flush_current()
    return [u for u in units if u["text"].strip()]


def split_roman_paragraph_unit_into_subunits(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    if unit["section_type"] != "body":
        return [dict(
            unit,
            subunit_label=None,
            subunit_title=None,
            roman_subunit_label=None,
            roman_subunit_title=None,
        )]

    text = unit["text"]
    matches = list(NUMBERED_ITEM_START_RE.finditer(text))

    # No numbered children -> keep section as one semantic unit
    if not matches:
        return [dict(
            unit,
            subunit_label=None,
            subunit_title=None,
            roman_subunit_label=None,
            roman_subunit_title=None,
        )]

    parent_prefix = build_parent_heading_prefix(unit)
    out: List[Dict[str, Any]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        label = match.group(1).strip()
        title = clean_subunit_title(match.group(2).strip())

        if parent_prefix and not block.startswith(parent_prefix):
            block = f"{parent_prefix}\n{block}"

        out.append({
            **unit,
            "subunit_label": label,
            "subunit_title": title,
            "roman_subunit_label": None,
            "roman_subunit_title": None,
            "text": block.strip(),
        })

    return out


def split_roman_children_with_nested_roman(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    numbered_children = split_roman_paragraph_unit_into_subunits(unit)
    final_units: List[Dict[str, Any]] = []

    for child in numbered_children:
        matches = list(ROMAN_SUBUNIT_RE.finditer(child["text"]))

        if not matches:
            final_units.append(child)
            continue

        # If there are nested roman markers, split them too
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(child["text"])
            block = child["text"][start:end].strip()

            final_units.append({
                **child,
                "roman_subunit_label": match.group(1).lower(),
                "roman_subunit_title": clean_subunit_title(match.group(2).strip()),
                "text": block,
            })

    return final_units


# ---------- Parser 3: paragraph fallback ----------

def split_paragraph_fallback_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    text = "\n\n".join(page for _, page in pages)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    units = []
    for i, para in enumerate(paragraphs, start=1):
        units.append({
            "section_type": "body",
            "section_number": None,
            "section_title": f"Paragraph {i}",
            "text": para,
            "page_numbers": [],
        })
    return units


# ---------- Semantic unit builder ----------

def split_appendix_into_blocks(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    if unit["section_type"] != "appendix":
        return [unit]

    text = unit["text"]
    lines = text.splitlines()

    blocks = []
    current_title = "Appendix"
    current_lines = []
    page_numbers = unit["page_numbers"]

    def flush():
        if current_lines:
            blocks.append({
                **unit,
                "section_number": None,
                "section_title": "Appendix",
                "subsection_number": None,
                "subsection_title": None,
                "subunit_label": None,
                "subunit_title": current_title,
                "roman_subunit_label": None,
                "roman_subunit_title": None,
                "text": "\n".join(current_lines).strip(),
                "page_numbers": page_numbers,
            })

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        # Keep main appendix heading in the first block
        if stripped == "## Appendix":
            current_lines.append(stripped)
            continue

        looks_like_appendix_block_title = (
            stripped.startswith("## ")
            or (
                len(stripped) <= 120
                and not stripped.startswith("-")
                and (
                    "Restricted Key Words" in stripped
                    or "Restricted Trademark Terms" in stripped
                    or re.match(
                        r"^[A-Z][A-Za-z0-9/&'\"(). -]{2,80}$",
                        stripped
                    )
                )
            )
        )

        if looks_like_appendix_block_title and current_lines:
            flush()
            current_title = stripped.replace("## ", "").strip()
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    flush()
    return blocks if blocks else [unit]


def count_roman_paragraph_markers(text: str) -> int:
    return len(ROMAN_SECTION_RE.findall(text)) + len(PARAGRAPH_SECTION_RE.findall(text))


def build_semantic_units(markdown_text: str, source_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pages = split_pages(markdown_text)
    base_metadata = detect_metadata_from_pages(pages)
    full_text = "\n\n".join(page for _, page in pages)

    article_score = count_article_section_markers(full_text)
    numeric_score = count_numeric_section_markers(full_text)
    roman_score = count_roman_paragraph_markers(full_text)

    semantic_units: List[Dict[str, Any]]

    if article_score >= 3:
        top_units = split_article_section_units(pages)
        semantic_units = []
        for unit in top_units:
            semantic_units.extend(split_article_section_into_subunits(unit))
    elif roman_score >= 2:
        top_units = split_roman_paragraph_units(pages)
        semantic_units = []
        for unit in top_units:
            semantic_units.extend(split_roman_paragraph_unit_into_subunits(unit))       
    elif numeric_score >= 2:
        top_units = split_numeric_top_sections(pages)
        semantic_units = []
        for unit in top_units:
            if unit["section_type"] == "appendix":
                semantic_units.extend(split_appendix_into_blocks(unit))
            else:
                semantic_units.extend(split_numeric_section_into_subsections(unit))
    else:
        semantic_units = split_paragraph_fallback_units(pages)

    for i, unit in enumerate(semantic_units, start=1):
        stem = Path(source_file).stem
        unit["semantic_unit_id"] = f"{stem}|unit|{i}"
        unit["source"] = source_file
        unit["semantic_unit_token_count"] = count_tokens(unit["text"])

    return semantic_units, base_metadata


# ---------- Final child-chunk builder ----------

def split_paragraph_by_tokens(paragraph: str, max_tokens: int, overlap_tokens: int) -> List[str]:
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


def split_semantic_unit_to_child_chunks(text: str, max_tokens: int = 256, overlap_tokens: int = 40) -> List[str]:
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
            child_chunks.extend(split_paragraph_by_tokens(para, max_tokens, overlap_tokens))
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
    base_metadata: Dict[str, Any],
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

        child_ids = [f"{unit['semantic_unit_id']}|chunk|{i+1}" for i in range(len(child_texts))]

        for i, child_text in enumerate(child_texts):
            chunk_id = child_ids[i]
            page_numbers = unit.get("page_numbers") or []
            docs.append(Document(
                id=chunk_id,
                page_content=child_text,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": chunk_id,
                    "document_id": base_metadata.get("document_id"),
                    "semantic_unit_id": unit["semantic_unit_id"],
                    "source": unit["source"],
                    "source_path": base_metadata.get("source_path"),
                    "file_name": base_metadata.get("file_name"),
                    "dataset": base_metadata.get("dataset"),
                    "corpus": base_metadata.get("corpus"),
                    "part": base_metadata.get("part"),
                    "contract_type": base_metadata.get("contract_type"),
                    "document_title": base_metadata.get("document_title"),
                    "title": base_metadata.get("title"),
                    "company_names": base_metadata.get("company_names"),
                    "all_companies": base_metadata.get("all_companies"),
                    "party_count": base_metadata.get("party_count"),
                    "section_type": unit.get("section_type"),
                    "article_number": unit.get("article_number"),
                    "article_title": unit.get("article_title"),
                    "section_number": unit.get("section_number"),
                    "section_title": unit.get("section_title"),
                    "roman_section_number": unit.get("roman_section_number"),
                    "roman_section_title": unit.get("roman_section_title"),
                    "subsection_number": unit.get("subsection_number"),
                    "subsection_title": unit.get("subsection_title"),
                    "subunit_label": unit.get("subunit_label"),
                    "subunit_title": unit.get("subunit_title"),
                    "roman_subunit_label": unit.get("roman_subunit_label"),
                    "roman_subunit_title": unit.get("roman_subunit_title"),
                    "page_numbers": page_numbers,
                    "page_start": page_numbers[0] if page_numbers else None,
                    "page_end": page_numbers[-1] if page_numbers else None,
                    "semantic_unit_token_count": unit.get("semantic_unit_token_count"),
                    "token_count": count_tokens(child_text),
                    "child_chunk_index": i + 1,
                    "child_chunk_count": len(child_texts),
                    "prev_chunk_id": child_ids[i - 1] if i > 0 else None,
                    "next_chunk_id": child_ids[i + 1] if i < len(child_ids) - 1 else None,
                }
            ))

    return docs


# ---------- Loader ----------

def load_pdf(
    relative_path: str,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
    debug: bool = False,
) -> List[Document]:
    root_path = Path(__file__).resolve().parents[3]
    pdf_path = root_path / Path(relative_path)
    assert pdf_path.exists(), f"File not found: {pdf_path}"

    loader = DoclingLoader(
        file_path=str(pdf_path),
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={
            "page_break_placeholder": f"\n\n{PAGE_BREAK}\n\n",
        },
    )

    # for doc in loader.load():
    #     print(doc.page_content)

    raw_docs = loader.load()
    if not raw_docs:
        return []

    markdown_text = raw_docs[0].page_content
    semantic_units, base_metadata = build_semantic_units(markdown_text, str(relative_path))
    document_metadata = build_document_metadata(base_metadata, str(relative_path), semantic_units)

    docs = build_final_documents(
        semantic_units=semantic_units,
        base_metadata=document_metadata,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if debug:
        print(f"Built {len(semantic_units)} semantic units and {len(docs)} chunks.")
        for i, unit in enumerate(semantic_units, start=1):
            print("-" * 80)
            print(f"Unit {i}")
            print(f"Document Title: {document_metadata.get('document_title')}")
            print(f"Company Names: {document_metadata.get('company_names')}")
            print(f"Party Count: {document_metadata.get('party_count')}")
            print(f"Source: {unit['source']}")
            print(f"Semantic Unit ID: {unit['semantic_unit_id']}")
            print(f"Article: {unit.get('article_number')} - {unit.get('article_title')}")
            print(f"Section: {unit.get('section_number')} - {unit.get('section_title')}")
            print(f"Roman Section: {unit.get('roman_section_number')} - {unit.get('roman_section_title')}")
            print(f"Subsection: {unit.get('subsection_number')} - {unit.get('subsection_title')}")
            print(f"Roman Subunit: {unit.get('roman_subunit_label')} - {unit.get('roman_subunit_title')}")
            print(f"Subunit: {unit.get('subunit_label')} - {unit.get('subunit_title')}")
            print(f"Token Count: {unit['semantic_unit_token_count']}")
            print(unit["text"])

    return docs


def iter_corpus_pdf_paths(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> List[str]:
    root_path = Path(__file__).resolve().parents[3]
    corpus_path = root_path / Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    pdf_paths = [
        path.relative_to(root_path).as_posix()
        for path in corpus_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(pdf_paths)


def load_corpus(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
    max_documents: Optional[int] = None,
    debug: bool = False,
) -> List[Document]:
    documents: List[Document] = []
    pdf_paths = iter_corpus_pdf_paths(corpus_dir)

    if max_documents is not None:
        pdf_paths = pdf_paths[:max_documents]

    for relative_path in pdf_paths:
        documents.extend(
            load_pdf(
                relative_path=relative_path,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                debug=debug,
            )
        )

    return documents