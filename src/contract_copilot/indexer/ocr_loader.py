import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from transformers import AutoTokenizer
from langchain_core.documents import Document
import pymupdf4llm


PAGE_BREAK = "<<<PAGE_BREAK>>>"
PYMUPDF_PAGE_END_RE = re.compile(
    r"(?im)^\s*---\s*end of page\.page_number=(\d+)\s*---\s*$"
)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_DIR = Path("data/raw/CUAD_v1/full_contract_pdf")


# ---------- Regexes ----------

# Parser experiment: CUAD has contracts with plain numeric headings like
# "1. Term". If PyMuPDF4LLM heading markdown is reliable enough, try disabling
# this parser path and compare chunk boundaries.
TOP_SECTION_RE = re.compile(
    r"(?m)^(?:##\s*)?(\d{1,2})\.\s+(?!\d)([^\n]{1,200})$"
)

# Parser experiment: detects "ARTICLE I" + "Section 1.1 Title" contracts.
# This is format-specific legal-structure logic, not generic cleanup.
ARTICLE_RE = re.compile(
    r"(?m)^(?:##\s*)?ARTICLE\s+([IVXLC]+)(?:\s+(.{1,140}?))?\s*$",
    re.IGNORECASE,
)
SECTION_XY_RE = re.compile(
    r"(?im)^(?:##\s*)?Section\s+(\d+\.\d+)\s+(.+)$"
)

# Parser experiment: detects German/European-style structures such as
# "## I. General" followed by "§ 1 Scope" or "A. Definitions".
# Disable this path if PyMuPDF4LLM now gives more consistent headings.
ROMAN_SECTION_RE = re.compile(
    r"(?im)^##\s*([IVXLC]+)\.\s+(.{1,120}?)\s*$"
)
PARAGRAPH_SECTION_RE = re.compile(
    r"(?im)^(?:##\s*)?§\s*(\d+(?:\.\d+)*)\s+(.{1,160}?)(?=\s*\(\d+\)\s+|\.\s*$|$)"
)
LETTER_SECTION_RE = re.compile(
    r"(?im)^(?:##\s*)?([A-Z])\.\s+(.{1,160})$"
)

# Parser experiment: optional fine-grained splits inside long sections.
# These are intentionally conservative but can over-split body text if current
# parser output already provides good paragraph/chunk boundaries.
LETTER_SUBUNIT_RE = re.compile(
    r"(?im)^\(([a-z])\)\s+([^.\n]{1,100})(?:\.)?(?:\s|$)"
)
ROMAN_SUBUNIT_RE = re.compile(
    r"(?im)^\(((?:ix|iv|v?i{1,3}|x))\)\s+([^.\n:;]{1,120})(?:[.:;])?(?:\s|$)"
)
NUMBERED_ITEM_START_RE = re.compile(
    r"(?im)^\((\d+)\)\s+(.{1,120}?)(?=\.\s|:\s|$)"
)
SUBSECTION_MARKER_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<marker>[-+*])\s+)?"
    r"(?P<label>(?P<parent>\d{1,2}|[A-Z])\.\d+(?:\.\d+)*)\b"
    r"(?P<rest>.*)$"
)
LETTER_LIST_ITEM_RE = re.compile(r"\(([a-z])\)\s+", re.IGNORECASE)

ENTITY_SUFFIXES = (
    r'Inc\.?|LLC|L\.L\.C\.|Corp\.?|Corporation|Company|Ltd\.?|Limited|'
    r'LP|L\.P\.|LLP|L\.L\.P\.|PLC|Bank|N\.A\.|National Association'
)
# Parser experiment: company extraction is tuned to legal intro blocks and
# entity suffixes. It only affects metadata, so it is safe to compare with this
# disabled if file-name metadata is enough.
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

# Parser experiment: appendix handling is specific to CUAD exhibits/appendices.
APPENDIX_HEADING_RE = re.compile(r"(?im)^##\s*Appendix\s*$|^Appendix\s*$")
ATTACHMENT_PREFACE_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:Schedule|Exhibit|Annex|Attachment|Addendum)\s+[A-Z0-9]+[A-Z0-9.-]*\s*$",
    re.IGNORECASE,
)
ATTACHMENT_START_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?P<kind>Appendix|Schedule|Exhibit|Annex|Attachment|Addendum)"
    r"\s+(?P<label>[A-Z0-9]+[A-Z0-9.-]*)\b",
    re.IGNORECASE,
)
APPENDIX_CATEGORY_LABELS = (
    "Chase Brand",
    "AARP",
    "Amazon",
    "Borders",
    "Waldenbooks",
    "British Air",
    "Continental",
    "Disney",
    "Hess",
    "Holiday Inn/Priority Club",
    "Marathon",
    "Marriott",
    "Overstock",
    "Sony",
    "Speedway",
    "Starbucks",
    "Subaru",
    "Toys",
    "Trump",
    "United",
    "Universal",
    "Volkswagen",
    "UAL",
)
COMMON_TYPO_CORRECTIONS = {
    "UNIVERSTIY": "UNIVERSITY",
    "UNVIERSAL": "UNIVERSAL",
    "UNTIED": "UNITED",
    "UNTIEDAIR": "UNITEDAIR",
    "UNITIED": "UNITED",
}
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

    # Parser experiment: legacy cleanup for party names swallowed from noisy
    # introductions. Try removing if PyMuPDF4LLM text makes party extraction
    # stable without these tail trims.
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
    # Parser experiment: this targets older parser/source banners and standalone
    # page numbers. PyMuPDF4LLM may make the source-line removal unnecessary.
    # Remove parser/source lines
    text = re.sub(
        r'(?m)^Source:\s+.*?<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>\s*',
        '',
        text,
    )

    # Remove standalone page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    return text


def normalize_markdown_heading_markup(text: str) -> str:
    """
    PyMuPDF4LLM may wrap bold headings in Markdown emphasis markers.
    Unwrap whole-line emphasis so structural regexes see the heading text,
    while preserving inline emphasis inside normal prose.
    """
    normalized_lines = []
    for line in text.splitlines():
        line = re.sub(
            r"^(\s*#{1,6}\s*)?(\*\*|__|\*|_)(\S(?:.*?\S)?)\2\s*$",
            r"\1\3",
            line,
        )
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def normalize_inline_markdown_emphasis(text: str) -> str:
    """
    Remove inline Markdown emphasis markers emitted by the parser while keeping
    the emphasized content.
    """
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<!\*)\*\*([^\n*]+?)\*\*(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)__([^\n_]+?)__(?!_)", r"\1", text)

    return text


def normalize_quoted_text_spacing(text: str) -> str:
    return re.sub(r'"[ \t]+([^"\n]{1,120}?\S)[ \t]+"', r'"\1"', text)


def normalize_missing_punctuation_spacing(text: str) -> str:
    """
    Repair parser-glued punctuation without touching common numeric/legal
    patterns such as 19.3, 1,000, or U.S.
    """
    text = re.sub(r'([,;:])(?=(?:"|[A-Za-z]))', r'\1 ', text)
    text = re.sub(r'(?<=[a-z0-9)\]"])\.(?=(?:"[A-Z])|[A-Z])', r'. ', text)
    text = re.sub(r'(?<=[A-Za-z0-9.!?])"(?=[A-Za-z])', r'" ', text)
    return text


def normalize_quoted_section_headings(text: str) -> str:
    """
    PyMuPDF sometimes emits section headings with a stray leading quote instead
    of Markdown heading markup, e.g. "' Section 13.1 Confidential Treatment."
    """
    return re.sub(
        r"(?im)^\s*['\"]\s*(Section\s+\d+(?:\.\d+)*\b)",
        r"## \1",
        text,
    )


def normalize_heading_paragraph_breaks(text: str) -> str:
    heading_start = (
        r"(?:#{1,6}\s*)?"
        r"(?:ARTICLE\s+[IVXLC]+|Section\s+\d+(?:\.\d+)*|§\s*\d+(?:\.\d+)*|\d{1,2}\.\s+\D)"
    )
    return re.sub(rf"(?im)([^\n])\n({heading_start})", r"\1\n\n\2", text)


def normalize_inline_article_heading_breaks(text: str) -> str:
    return re.sub(
        r"(?i)([.;])\s+(ARTICLE\s+[IVXLC]+\b(?:\s+[A-Z][A-Z ]{1,80})?)",
        r"\1\n\n## \2",
        text,
    )


def promote_paragraph_section_headings(text: str) -> str:
    # Parser experiment: promotes bare "§ 1 Title" lines into markdown headings.
    # Try disabling if PyMuPDF4LLM already emits reliable markdown headings.
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
    # Parser experiment: repair step for headings/list items glued onto prose.
    # With cleaner PyMuPDF4LLM output, this may be the first normalization to
    # test without.
    # ARTICLE headings
    text = re.sub(
        r'(?<!\n)(?<!## )(ARTICLE\s+[IVXLC]+\b)',
        r'\n\1',
        text,
        flags=re.IGNORECASE,
    )

    # Section X.Y headings
    text = re.sub(
        r'(?<!\n)(?<!## )(Section\s+\d+\.\d+\s+)',
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


def normalize_missing_space_after_number_period(text: str) -> str:
    """
    Fix cases like:
      ## 5.Term of this Agreement   -> ## 5. Term of this Agreement
      13.5Governing Law             -> 13.5 Governing Law   (optional)
      § 4Independence               -> § 4 Independence     (optional)
    """

    # Parser experiment: legacy repair for glued OCR headings. If the new parser
    # rarely emits "5.Term" or "13.5Governing", disable this whole function.
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
    # Parser experiment: repairs parser-created false headings such as
    # "## 13. 5 Governing" back into "13.5 Governing".
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
    # Parser experiment: another OCR/parser repair for split subsection numbers
    # like "13. 5". Disable if PyMuPDF4LLM no longer produces this shape.
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
    # Parser experiment: section-sign repair is highly format-specific. It helps
    # with glued "§" headings, but may be unnecessary with cleaner parsing.
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
    # Parser experiment: this can improve child splitting for "(1)" clauses,
    # but it is a legacy glue repair. Test without it on current PyMuPDF4LLM
    # output.
    # If a numbered item appears after prose, split it to a new line
    text = re.sub(r'(?<!\n)\s(\((?:\d+)\)\s+)', r'\n\1', text)

    return text


def split_flattened_dash_list_line(line: str) -> List[str] | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None

    parts = [part.strip() for part in re.split(r"\s+-\s+", stripped[2:]) if part.strip()]
    if len(parts) < 3:
        return None

    for part in parts:
        word_count = count_words(part)
        if word_count == 0 or word_count > 8:
            return None
        if re.search(r"[,;:!?]", part):
            return None

    indent = re.match(r"^(\s*)", line).group(1)
    return [f"{indent}- {part}" for part in parts]


def parenthetical_letter_index(label: str) -> int:
    return ord(label.lower()) - ord("a")


def split_flattened_parenthetical_letter_list_line(line: str) -> List[str] | None:
    match = re.match(r"^(?P<indent>\s*)(?:(?P<marker>[-+*])\s+)?(?=\([a-z]\)\s+)", line, re.IGNORECASE)
    if not match:
        return None

    body_start = match.end()
    body = line[body_start:]
    matches = list(LETTER_LIST_ITEM_RE.finditer(body))
    if len(matches) < 2:
        return None

    labels = [item.group(1).lower() for item in matches]
    label_indexes = [parenthetical_letter_index(label) for label in labels]
    if label_indexes != sorted(label_indexes) or len(set(label_indexes)) != len(label_indexes):
        return None

    expected_first = label_indexes[0]
    if label_indexes != list(range(expected_first, expected_first + len(label_indexes))):
        return None

    indent = "" if match.group("marker") else match.group("indent")
    prefix = f"{indent}{match.group('marker') or '-'} "
    parts = []
    for index, item in enumerate(matches):
        start = item.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        part = body[start:end].strip()
        if part:
            parts.append(f"{prefix}{part}")

    return parts or None


def normalize_flattened_list_items(text: str) -> str:
    """
    Repair parser-flattened list rows such as:
      - TERM ONE - TERM TWO - TERM THREE
      - (a) First item (b) Second item (c) Third item
    into one bullet per item. The guard is intentionally narrow so normal legal
    prose that happens to contain list-like tokens is left alone.
    """
    line_splitters = (
        split_flattened_dash_list_line,
        split_flattened_parenthetical_letter_list_line,
    )
    normalized_lines = []

    for line in text.splitlines():
        split_lines = next(
            (
                candidate
                for splitter in line_splitters
                if (candidate := splitter(line))
            ),
            None,
        )
        if split_lines:
            for index, split_line in enumerate(split_lines):
                if index > 0:
                    normalized_lines.append("")
                normalized_lines.append(split_line)
        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def dominant_subsection_marker(marker_counts: Dict[str, int]) -> str | None:
    if not marker_counts:
        return None

    preferred_order = {"-": 0, "+": 1, "*": 2}
    return sorted(
        marker_counts.items(),
        key=lambda item: (-item[1], preferred_order.get(item[0], 99)),
    )[0][0]


def normalize_missing_subsection_markers(text: str) -> str:
    """
    Restore a missing list marker on a subsection line when sibling subsection
    lines with the same parent label consistently use one.
    """
    marker_counts_by_parent: Dict[str, Dict[str, int]] = {}

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue

        match = SUBSECTION_MARKER_LINE_RE.match(line)
        if not match or not match.group("marker"):
            continue

        parent = match.group("parent")
        marker = match.group("marker")
        marker_counts = marker_counts_by_parent.setdefault(parent, {})
        marker_counts[marker] = marker_counts.get(marker, 0) + 1

    marker_by_parent = {
        parent: marker
        for parent, marker_counts in marker_counts_by_parent.items()
        if (marker := dominant_subsection_marker(marker_counts))
    }
    if not marker_by_parent:
        return text

    normalized_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            normalized_lines.append(line)
            continue

        match = SUBSECTION_MARKER_LINE_RE.match(line)
        if not match or match.group("marker"):
            normalized_lines.append(line)
            continue

        marker = marker_by_parent.get(match.group("parent"))
        if not marker:
            normalized_lines.append(line)
            continue

        normalized_lines.append(
            f"{match.group('indent')}{marker} {match.group('label')}{match.group('rest')}"
        )

    return "\n".join(normalized_lines)


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
    text = re.sub(r'\(\?([^?\n]{1,80})\?\)', r'("\1")', text)
    text = re.sub(r'\[([A-Za-z0-9.-]+\.[A-Za-z]{2,})\]', r'\1', text)
    text = re.sub(r'\bLinkshare NetworkTM\b', 'Linkshare Network™', text)

    text = strip_parser_artifacts(text)
    text = normalize_markdown_heading_markup(text)
    text = normalize_inline_markdown_emphasis(text)
    text = normalize_quoted_text_spacing(text)
    text = normalize_missing_punctuation_spacing(text)
    text = normalize_quoted_section_headings(text)
    text = normalize_inline_article_heading_breaks(text)
    text = normalize_heading_paragraph_breaks(text)
    text = normalize_flattened_list_items(text)
    text = normalize_missing_subsection_markers(text)
    # Parser experiment disabled: old heading/list break repair.
    # text = inject_heading_breaks(text)
    # text = normalize_section_sign_headings(text)
    # text = inject_numbered_item_breaks(text)

    # Parser experiment disabled: old subsection-number repair.
    # text = normalize_false_markdown_subsection_headers(text)
    # text = normalize_broken_subsection_numbers(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\(\s*([a-zivx\d]+)\s*\)", r"(\1)", text, flags=re.IGNORECASE)

    text = re.sub(r"(?m)^Appendix\s*$", "## Appendix", text)

    return text.strip()


def split_pages(markdown_text: str) -> List[Tuple[int, str]]:
    if PYMUPDF_PAGE_END_RE.search(markdown_text):
        pages = []
        current_page_no = 1
        current_start = 0

        for match in PYMUPDF_PAGE_END_RE.finditer(markdown_text):
            page = normalize_text(markdown_text[current_start:match.start()])
            if page.strip():
                pages.append((current_page_no, page.strip()))

            current_page_no = int(match.group(1)) + 1
            current_start = match.end()

        page = normalize_text(markdown_text[current_start:])
        if page.strip():
            pages.append((current_page_no, page.strip()))

        return pages

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


def infer_title_from_front_matter(front_matter_text: str | None) -> str | None:
    if not front_matter_text:
        return None

    for line in front_matter_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        title = stripped.lstrip("#").strip()
        if len(title.split()) <= 12 and any(char.isalpha() for char in title):
            return title.title()

    for line in front_matter_text.splitlines():
        title = line.strip().lstrip("#").strip()
        if not title:
            continue
        if len(title.split()) <= 12 and any(char.isalpha() for char in title):
            return title.title()

    return None


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

    # Parser experiment disabled: intro/company regex extraction is tuned to
    # older noisy parser output. Re-enable this block if filename metadata is
    # not enough.
    # first_text = next(
    #     (
    #         unit["text"][:2000]
    #         for unit in semantic_units
    #         if unit.get("section_type") in {"front_matter", "body"} and unit.get("text", "").strip()
    #     ),
    #     "",
    # )
    # companies = dedupe_companies(extract_all_companies_from_intro(first_text))
    companies = []
    if not companies:
        inferred_company = infer_company_from_filename(source_file)
        if inferred_company:
            companies = [inferred_company]

    front_matter_title = infer_title_from_front_matter(base_metadata.get("front_matter_text"))
    document_title = (
        base_metadata.get("document_title")
        or front_matter_title
        or prettify_title(source_metadata["document_id"])
    )

    return {
        **base_metadata,
        **source_metadata,
        "document_title": document_title,
        "title": document_title,
        "company_names": companies,
        "all_companies": companies,
        "party_count": len(companies),
        "front_matter_text": base_metadata.get("front_matter_text"),
        "front_matter_pages": base_metadata.get("front_matter_pages"),
    }


# ---------- Format detection ----------

def count_article_section_markers(text: str) -> int:
    return len(ARTICLE_RE.findall(text)) + len(SECTION_XY_RE.findall(text))


def should_use_article_section_parser(text: str) -> bool:
    article_count = len(ARTICLE_RE.findall(text))
    section_count = len(SECTION_XY_RE.findall(text))
    return article_count >= 2 and section_count >= 3


def count_numeric_section_markers(text: str) -> int:
    return len(TOP_SECTION_RE.findall(text))


# ---------- Parser 1: numeric section format ----------

def make_subsection_regex_for_section(section_number: str) -> re.Pattern:
    # Parser experiment: numeric subsection detection is anchored to the current
    # parent section, e.g. only "13.x" inside section "13". This avoids many
    # false positives, but can be skipped if token chunking is enough.
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
    # Parser experiment: title inference is deliberately heuristic. Disable
    # title extraction first, before disabling subsection splitting, if metadata
    # titles look noisy.
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
    # Parser experiment: removes duplicated parent headings that were injected
    # for retrieval context before child splits. If heading duplication is gone,
    # this helper may no longer be needed.
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
    # Parser experiment: shared splitter for "(a)" and "(i)" markers. It is
    # intentionally format-specific and may be too granular for cleaner parser
    # output.
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


def clean_section_heading_title(title: str, max_words: int = 14) -> str:
    title = " ".join(title.split()).strip()
    match = re.match(r"(.{1,160}?)[.:]\s+[A-Z]", title)
    if match:
        title = match.group(1)
    return clean_heading_title(title, max_words=max_words)


def clean_subunit_title(title: str, max_words: int = 10) -> str:
    title = " ".join(title.split()).strip(" .;,:")
    words = title.split()
    return " ".join(words[: max_words])


def split_article_section_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    # Parser experiment: parser for ARTICLE / Section X.Y contracts. Try
    # bypassing this branch in build_semantic_units if PyMuPDF4LLM headings are
    # already sufficient for simple paragraph/token chunking.
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

    def article_heading(number: str | None, title: str | None) -> str | None:
        if not number:
            return None
        return f"## ARTICLE {number}{f' {title}' if title else ''}".strip()

    def is_article_heading_only_current() -> bool:
        if current.get("section_number") is not None:
            return False

        heading = article_heading(current.get("article_number"), current.get("article_title"))
        if not heading:
            return False

        content_parts = [part.strip() for part in current["content_parts"] if part.strip()]
        return content_parts == [heading]

    def flush_current() -> None:
        nonlocal current
        if not current["content_parts"]:
            return

        if not is_article_heading_only_current():
            units.append({
                **current,
                "text": "\n".join(current["content_parts"]).strip(),
                "page_numbers": sorted(set(current["page_numbers"])),
            })

        current = {
            "section_type": "front_matter",
            "article_number": current_article_number,
            "article_title": current_article_title,
            "section_number": None,
            "section_title": current_article_title or "Article Boundary",
            "content_parts": [],
            "page_numbers": [],
        }

    for page_no, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                if current["content_parts"] and current["content_parts"][-1] != "":
                    current["content_parts"].append("")
                continue

            m_article = ARTICLE_RE.match(line)
            if m_article:
                flush_current()

                current_article_number = m_article.group(1).upper()
                current_article_title = clean_heading_title(m_article.group(2).strip()) if m_article.group(2) else None
                pending_article_title = current_article_title is None
                heading = article_heading(current_article_number, current_article_title)
                current = {
                    "section_type": "body",
                    "article_number": current_article_number,
                    "article_title": current_article_title,
                    "section_number": None,
                    "section_title": current_article_title or f"Article {current_article_number}",
                    "content_parts": [heading] if heading else [],
                    "page_numbers": [page_no],
                }
                continue

            if pending_article_title and line.isupper() and len(line) <= 120:
                current_article_title = clean_heading_title(line)
                pending_article_title = False
                current["article_title"] = current_article_title
                current["section_title"] = current_article_title
                current["content_parts"] = [article_heading(current_article_number, current_article_title)]
                current["page_numbers"].append(page_no)
                continue
            else:
                pending_article_title = False

            m_attachment = ATTACHMENT_START_RE.match(line.lstrip("#").strip())
            if m_attachment:
                flush_current()
                title = attachment_title_from_match(m_attachment)
                current = {
                    "section_type": "appendix",
                    "article_number": None,
                    "article_title": None,
                    "section_number": None,
                    "section_title": title,
                    "attachment_type": m_attachment.group("kind").lower(),
                    "attachment_label": m_attachment.group("label").upper(),
                    "content_parts": [promote_semantic_heading_markup(line)],
                    "page_numbers": [page_no],
                }
                continue

            m_section = SECTION_XY_RE.match(line)
            if m_section:
                flush_current()

                sec_num = m_section.group(1).strip()
                sec_title = clean_section_heading_title(m_section.group(2).strip())

                header_parts = []
                heading = article_heading(current_article_number, current_article_title)
                if heading:
                    header_parts.append(heading)
                    header_parts.append("")
                header_parts.append(f"## Section {sec_num} {m_section.group(2).strip()}")

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

    flush_current()

    return [u for u in units if u["text"].strip()]


def split_article_section_into_subunits(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Parser experiment: splits legal clauses into "(a)" and nested "(i)" units.
    # This improves focused retrieval on long sections, but can create tiny or
    # awkward chunks if the source markdown is already well segmented.
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


def is_heading_only_unit(unit: Dict[str, Any]) -> bool:
    text = unit["text"].strip()
    if unit.get("roman_section_number") and not unit.get("section_number"):
        heading = f"## {unit['roman_section_number']}. {unit['roman_section_title']}".strip()
        return text == heading
    return False


def split_roman_paragraph_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    # Parser experiment: parser for Roman numeral / paragraph-sign contracts.
    # This is one of the more specialized branches and is worth testing without
    # on the new parser output.
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
    # Parser experiment: splits "§" sections into "(1)", "(2)" children. It may
    # be unnecessary if PyMuPDF4LLM emits natural paragraph boundaries.
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
    # Parser experiment: currently unused nested "(i)" splitter for Roman-style
    # sections. Keep only if you decide nested roman metadata is useful.
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

def looks_like_list_item_block(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(re.match(
        r"^(?:[-+*]\s+|\d+[.)]\s+|\([A-Za-z0-9ivxlcdmIVXLCDM]+\)\s+)",
        first_line,
    ))


def looks_like_heading_block(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(
        first_line.startswith("#")
        or re.match(r"^(?:ARTICLE\s+[IVXLC]+|Section\s+\d+(?:\.\d+)*|§\s*\d+)", first_line, re.IGNORECASE)
        or re.match(r"^\d{1,2}\.\s+\D", first_line)
    )


def looks_like_section_heading_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    line = lines[0]
    return bool(
        re.match(r"^#{1,6}\s*\d{1,2}\.\s*\D", line)
        or re.match(r"^\d{1,2}\.\s+\D", line)
        or re.match(r"^(?:#{1,6}\s*)?Section\s+\d+(?:\.\d+)*\b", line, re.IGNORECASE)
        or re.match(r"^(?:#{1,6}\s*)?§\s*\d+", line)
    )


def promote_semantic_heading_markup(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("#"):
        return stripped

    return f"## {stripped}"


def should_merge_page_continuation(previous_text: str, next_text: str) -> bool:
    if not previous_text.strip() or not next_text.strip():
        return False

    if looks_like_heading_block(next_text) or looks_like_list_item_block(next_text):
        return False

    previous_tail = previous_text.rstrip()
    if previous_tail.endswith((".", ":", ";", "?", "!", '"', "'")):
        return False

    next_start = next_text.lstrip()
    return bool(
        next_start
        and (
            next_start[0].islower()
            or next_start[0] in ",.;:)]}"
            or looks_like_orphan_sentence_tail(next_text)
        )
    )


def looks_like_continuation_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if looks_like_heading_block(stripped) or looks_like_list_item_block(stripped):
        return False

    return bool(
        re.match(r"^(?:https?://|www\.)", stripped, re.IGNORECASE)
        or re.match(r"^For\s+(?:new|existing)\s+affiliates?\s*:", stripped, re.IGNORECASE)
    )


def looks_like_orphan_sentence_tail(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if looks_like_heading_block(stripped) or looks_like_list_item_block(stripped):
        return False

    return count_words(stripped) <= 2 and stripped.endswith((".", ",", ";", ":", ")", "]"))


def looks_like_short_tail_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if looks_like_heading_block(stripped) or looks_like_list_item_block(stripped):
        return False

    return count_words(stripped) <= 2 and stripped.endswith((".", ",", ";", ":", ")", "]"))


def looks_like_form_control_block(text: str) -> bool:
    return text.strip().lower().strip(".:") in {"accept", "agree", "submit", "cancel"}


def append_to_semantic_unit(unit: Dict[str, Any], text: str, page_no: int, separator: str = "\n\n") -> None:
    unit["text"] = f"{unit['text'].rstrip()}{separator}{text.strip()}"
    unit["page_numbers"] = sorted(set([*unit.get("page_numbers", []), page_no]))


def split_paragraphs_with_list_items(text: str) -> List[str]:
    paragraphs: List[str] = []

    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if paragraphs and looks_like_list_item_block(para):
            paragraphs[-1] = f"{paragraphs[-1].rstrip()}\n\n{para}"
        else:
            paragraphs.append(para)

    return paragraphs


def split_paragraph_fallback_units(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    units = []
    paragraph_index = 1

    for page_no, page_text in pages:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

        for para in paragraphs:
            if looks_like_form_control_block(para):
                continue

            if looks_like_heading_block(para):
                units.append({
                    "section_type": "body",
                    "section_number": None,
                    "section_title": f"Paragraph {paragraph_index}",
                    "text": promote_semantic_heading_markup(para),
                    "page_numbers": [page_no],
                })
                paragraph_index += 1
                continue

            if units and looks_like_list_item_block(para):
                append_to_semantic_unit(units[-1], para, page_no)
                continue

            if units and looks_like_continuation_block(para):
                append_to_semantic_unit(units[-1], para, page_no)
                continue

            if units and looks_like_short_tail_block(para):
                append_to_semantic_unit(units[-1], para, page_no, separator=" ")
                continue

            if units and looks_like_section_heading_only(units[-1]["text"]):
                append_to_semantic_unit(units[-1], para, page_no)
                continue

            if (
                units
                and should_merge_page_continuation(units[-1]["text"], para)
            ):
                append_to_semantic_unit(units[-1], para, page_no, separator=" ")
                continue

            units.append({
                "section_type": "body",
                "section_number": None,
                "section_title": f"Paragraph {paragraph_index}",
                "text": para,
                "page_numbers": [page_no],
            })
            paragraph_index += 1

    return units


def is_numbered_section_heading_text(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(re.match(r"^#{1,6}\s*\d{1,2}\.\s+\D", first_line))


def is_body_section_heading_text(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(
        is_numbered_section_heading_text(text)
        or re.match(r"^#{1,6}\s*ARTICLE\s+[IVXLC]+\b", first_line, re.IGNORECASE)
        or re.match(r"^#{1,6}\s*Section\s+\d+(?:\.\d+)*\b", first_line, re.IGNORECASE)
        or re.match(r"^#{1,6}\s*§\s*\d+(?:\.\d+)*\b", first_line)
    )


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def looks_like_front_matter_unit(text: str, max_words: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if is_numbered_section_heading_text(stripped):
        return False

    if stripped.startswith("#"):
        return count_words(stripped) <= max_words

    if count_words(stripped) > max_words:
        return False

    if re.search(r"\b(?:agreement|party|affiliate|shall|will|must|may)\b", stripped, re.IGNORECASE):
        return False

    return True


def compact_front_matter_units(
    semantic_units: List[Dict[str, Any]],
    max_units: int = 6,
    max_words: int = 16,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    front_units = []
    body_start_index = 0

    for index, unit in enumerate(semantic_units[:max_units]):
        text = unit.get("text", "").strip()
        if not text:
            body_start_index = index + 1
            continue

        if not looks_like_front_matter_unit(text, max_words):
            body_start_index = index
            break

        front_units.append(unit)
        body_start_index = index + 1

    if not front_units:
        return semantic_units, {}

    front_text = "\n\n".join(unit["text"].strip() for unit in front_units if unit.get("text", "").strip())
    front_pages = sorted({
        page
        for unit in front_units
        for page in unit.get("page_numbers", [])
    })

    metadata = {
        "front_matter_text": front_text,
        "front_matter_pages": front_pages,
    }

    return semantic_units[body_start_index:], metadata


def merge_preamble_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    preamble_units = []
    body_start_index = 0

    for index, unit in enumerate(semantic_units):
        text = unit.get("text", "").strip()

        if is_body_section_heading_text(text):
            body_start_index = index
            break

        preamble_units.append(unit)
        body_start_index = index + 1

    if len(preamble_units) <= 1:
        return semantic_units

    merged = {
        **preamble_units[0],
        "section_title": "Preamble",
        "text": "\n\n".join(
            unit["text"].strip()
            for unit in preamble_units
            if unit.get("text", "").strip()
        ),
        "page_numbers": sorted({
            page
            for unit in preamble_units
            for page in unit.get("page_numbers", [])
        }),
    }

    return [merged, *semantic_units[body_start_index:]]


def starts_with_numbered_section_heading(text: str) -> bool:
    return is_numbered_section_heading_text(text)


def first_numbered_section(text: str) -> str | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = re.match(r"^#{1,6}\s*(\d{1,2})\.\s+\D", first_line)
    if match:
        return match.group(1)

    match = re.match(r"^(\d{1,2})\.\s+\D", first_line)
    return match.group(1) if match else None


def contains_subsection_for_parent(text: str, parent_number: str) -> bool:
    return bool(
        re.search(
            rf"(?m)(?:^|\n)\s*(?:[-+*]\s*)?{re.escape(parent_number)}\.\d+\b",
            text,
        )
    )


def looks_like_same_section_continuation(previous_text: str, text: str) -> bool:
    parent_number = first_numbered_section(previous_text)
    if not parent_number:
        return False

    next_section_number = first_numbered_section(text)
    if next_section_number and next_section_number != parent_number:
        return False

    return contains_subsection_for_parent(text, parent_number)


def merge_section_continuation_units(
    semantic_units: List[Dict[str, Any]],
    max_continuation_words: int = 220,
) -> List[Dict[str, Any]]:
    merged_units: List[Dict[str, Any]] = []

    for unit in semantic_units:
        text = unit.get("text", "").strip()

        if not merged_units:
            merged_units.append(unit)
            continue

        if unit.get("section_type") != "body":
            merged_units.append(unit)
            continue

        previous = merged_units[-1]
        if (
            previous.get("section_type") == "body"
            and starts_with_numbered_section_heading(previous.get("text", ""))
            and not looks_like_heading_block(text)
            and (
                count_words(text) <= max_continuation_words
                or unit.get("contains_list_continuation")
                or looks_like_same_section_continuation(previous.get("text", ""), text)
            )
        ):
            previous["text"] = f"{previous['text'].rstrip()}\n\n{text}"
            previous["page_numbers"] = sorted(set([
                *previous.get("page_numbers", []),
                *unit.get("page_numbers", []),
            ]))
            if unit.get("contains_list_continuation"):
                previous["contains_list_continuation"] = True
            continue

        merged_units.append(unit)

    return merged_units


def is_short_heading_only_text(text: str, max_words: int = 12) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    line = lines[0].lstrip("#").strip()
    return bool(line and count_words(line) <= max_words)


def looks_like_attachment_preface_heading(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    line = lines[0].lstrip("#").strip()
    if ATTACHMENT_PREFACE_HEADING_RE.match(lines[0]):
        return True

    words = re.findall(r"[A-Za-z]+", line)
    if not words or len(words) > 8:
        return False

    uppercase_words = sum(1 for word in words if word.isupper() or word.istitle())
    return uppercase_words == len(words)


def merge_attachment_preface_headings(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pending_headings: List[Dict[str, Any]] = []
    seen_attachment_heading = False

    def flush_pending() -> None:
        nonlocal seen_attachment_heading
        out.extend(pending_headings)
        pending_headings.clear()
        seen_attachment_heading = False

    for unit in semantic_units:
        text = unit.get("text", "").strip()

        if (
            is_short_heading_only_text(text)
            and (
                ATTACHMENT_PREFACE_HEADING_RE.match(text)
                or (seen_attachment_heading and looks_like_attachment_preface_heading(text))
            )
        ):
            pending_headings.append(unit)
            if ATTACHMENT_PREFACE_HEADING_RE.match(text):
                seen_attachment_heading = True
            continue

        if pending_headings and looks_like_heading_block(text):
            heading_text = "\n\n".join(
                pending["text"].strip()
                for pending in pending_headings
                if pending.get("text", "").strip()
            )
            unit = {
                **unit,
                "text": "\n\n".join(part for part in [heading_text, text] if part),
                "page_numbers": sorted({
                    page
                    for pending in pending_headings
                    for page in pending.get("page_numbers", [])
                } | set(unit.get("page_numbers", []))),
            }
            pending_headings.clear()
            seen_attachment_heading = False
            out.append(unit)
            continue

        if pending_headings:
            flush_pending()

        out.append(unit)

    if pending_headings:
        flush_pending()

    return out


LIST_CONTINUATION_CATALOG_RE = re.compile(
    r"(?is)(.*?\b(?:following|listed below)\s+(?:words|terms|items|phrases|keywords|names):\s*)"
    r"(\n\n-\s+.+)$"
)
def split_catalog_from_trailing_legal_bullets(catalog_text: str) -> Tuple[str, str | None]:
    bullet_blocks = re.split(r"(?=\n\n-\s+)", catalog_text)
    catalog_blocks = []
    trailing_blocks = []
    in_trailing_legal = False

    for block in bullet_blocks:
        stripped = block.strip()
        if not stripped:
            continue

        bullet_body = re.sub(r"^-\s+", "", stripped).strip()
        if re.match(
            r"^(?:Uses|Otherwise|Does|Is|Contains|Promotes|Engages|Misrepresents|Manipulates)\b",
            bullet_body,
            re.IGNORECASE,
        ):
            in_trailing_legal = True

        if in_trailing_legal:
            trailing_blocks.append(stripped)
        else:
            catalog_blocks.append(stripped)

    catalog = "\n\n".join(catalog_blocks).strip()
    trailing = "\n\n".join(trailing_blocks).strip()
    return catalog, trailing or None


def merge_list_continuation_catalog_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for unit in semantic_units:
        text = unit.get("text", "")
        match = LIST_CONTINUATION_CATALOG_RE.match(text)

        if not match:
            out.append(unit)
            continue

        lead_text = match.group(1).strip()
        catalog_text, trailing_legal_text = split_catalog_from_trailing_legal_bullets(match.group(2).strip())

        if count_words(catalog_text) < 40:
            out.append(unit)
            continue

        out.append({
            **unit,
            "contains_list_continuation": True,
            "text": "\n\n".join(
                part
                for part in [lead_text, catalog_text, trailing_legal_text]
                if part
            ),
        })

    return out


def split_list_continuation_catalog_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return merge_list_continuation_catalog_units(semantic_units)


def split_restricted_keyword_catalog_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return merge_list_continuation_catalog_units(semantic_units)


def looks_like_appendix_start(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(re.match(r"^#{1,6}\s*Appendix\b|^Appendix\b", first_line, re.IGNORECASE))


def looks_like_appendix_subheading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    if looks_like_appendix_start(first_line):
        return True

    if first_line.startswith("#"):
        return True

    if re.search(r"\b(?:Restricted|Trademark|Key Words|Terms)\b", first_line, re.IGNORECASE):
        return count_words(first_line) <= 8

    return False


def merge_appendix_units_by_heading(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    appendix_start = next(
        (
            index
            for index, unit in enumerate(semantic_units)
            if looks_like_appendix_start(unit.get("text", ""))
        ),
        None,
    )

    if appendix_start is None:
        return semantic_units

    prefix_units = semantic_units[:appendix_start]
    appendix_units = semantic_units[appendix_start:]
    grouped_appendix_units: List[Dict[str, Any]] = []
    header_parts = []
    header_pages = []

    for unit in appendix_units:
        text = unit.get("text", "").strip()
        if not text:
            continue

        if looks_like_appendix_subheading(text):
            header_parts.append(text)
            header_pages.extend(unit.get("page_numbers", []))
            continue

        full_text = "\n\n".join([*header_parts, text]).strip()
        first_line = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "Appendix")
        grouped_appendix_units.append({
            **unit,
            "section_type": "appendix",
            "section_title": clean_heading_title(first_line, max_words=8) or "Appendix",
            "text": full_text,
            "page_numbers": sorted(set([
                *header_pages,
                *unit.get("page_numbers", []),
            ])),
        })
        header_parts = []
        header_pages = []

    if header_parts:
        grouped_appendix_units.append({
            **appendix_units[-1],
            "section_type": "appendix",
            "section_title": "Appendix",
            "text": "\n\n".join(header_parts).strip(),
            "page_numbers": sorted(set(header_pages)),
        })

    return [*prefix_units, *grouped_appendix_units]


def appendix_category_pattern() -> re.Pattern:
    labels = sorted(APPENDIX_CATEGORY_LABELS, key=len, reverse=True)
    escaped = [re.escape(label) for label in labels]
    return re.compile(rf"(?<![A-Za-z0-9/&])({'|'.join(escaped)})(?=\s)")


def split_appendix_category_text(text: str) -> List[Tuple[str, str]]:
    header_lines = []
    body_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not body_lines and (
            not stripped
            or stripped.startswith("#")
            or re.search(r"\b(?:Restricted Trademark Terms|Partner Restricted Trademark Terms)\b", stripped, re.IGNORECASE)
        ):
            if stripped:
                header_lines.append(stripped)
            continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if not body:
        return [("Appendix", text.strip())] if text.strip() else []

    matches = []
    previous_label = None
    for match in appendix_category_pattern().finditer(body):
        label = match.group(1)
        if label == previous_label:
            continue
        matches.append(match)
        previous_label = label
    if not matches:
        title = clean_heading_title(body.splitlines()[0].strip(), max_words=8)
        return [(title or "Appendix", text.strip())]

    chunks: List[Tuple[str, str]] = []
    header = "\n\n".join(header_lines).strip()

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        chunk_body = body[start:end].strip(" ,\n")
        if not chunk_body:
            continue

        title = match.group(1)
        chunk_text = "\n\n".join(part for part in [header, chunk_body] if part)
        chunks.append((f"Appendix - {title}", chunk_text))

    return chunks or [("Appendix", text.strip())]


def split_appendix_units_by_category(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for unit in semantic_units:
        if unit.get("section_type") != "appendix":
            out.append(unit)
            continue

        chunks = split_appendix_category_text(unit.get("text", ""))
        if len(chunks) <= 1:
            title, text = chunks[0] if chunks else (unit.get("section_title", "Appendix"), unit.get("text", ""))
            out.append({
                **unit,
                "section_title": title or unit.get("section_title", "Appendix"),
                "text": text,
            })
            continue

        for title, text in chunks:
            out.append({
                **unit,
                "section_title": title,
                "text": text,
            })

    return out


def appendix_category_key(unit: Dict[str, Any]) -> str | None:
    if unit.get("section_type") != "appendix":
        return None

    title = unit.get("section_title", "")
    title = re.sub(r"^Appendix\s*-\s*", "", title, flags=re.IGNORECASE).strip()

    if title.upper() == "UAL":
        return "United"

    if title.lower().startswith("in addition"):
        return "Chase Brand"

    return title or None


def looks_like_appendix_cross_reference(unit: Dict[str, Any]) -> bool:
    if unit.get("section_type") != "appendix":
        return False

    text = " ".join(unit.get("text", "").split()).strip(" .")
    if count_words(text) > 6:
        return False

    return bool(
        re.fullmatch(
            r"[A-Z][A-Za-z/& ]{1,40}\s+See\s+[\"']?[A-Z][A-Za-z/& ]{1,40}[\"']?\s+Tab",
            text,
            flags=re.IGNORECASE,
        )
    )


def merge_related_appendix_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    appendix_key_to_index: Dict[str, int] = {}

    for unit in semantic_units:
        if looks_like_appendix_cross_reference(unit):
            continue

        key = appendix_category_key(unit)

        if not out or key is None:
            out.append(unit)
            continue

        if key in appendix_key_to_index:
            existing = out[appendix_key_to_index[key]]
            existing["text"] = f"{existing['text'].rstrip()}\n\n{unit['text'].strip()}"
            existing["page_numbers"] = sorted(set([
                *existing.get("page_numbers", []),
                *unit.get("page_numbers", []),
            ]))
            existing["section_title"] = f"Appendix - {key}"
            continue

        appendix_key_to_index[key] = len(out)
        out.append(unit)

    return out


def attachment_start_match(unit: Dict[str, Any]) -> re.Match | None:
    first_line = next(
        (line.strip().lstrip("#").strip() for line in unit.get("text", "").splitlines() if line.strip()),
        "",
    )
    return ATTACHMENT_START_RE.match(first_line)


def attachment_title_from_match(match: re.Match) -> str:
    kind = match.group("kind").title()
    label = match.group("label").upper()
    return f"{kind} {label}"


def merge_attachment_units(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            out.append(current)
            current = None

    for unit in semantic_units:
        match = attachment_start_match(unit)
        if match:
            flush_current()
            title = attachment_title_from_match(match)
            current = {
                **unit,
                "section_type": "appendix",
                "section_title": title,
                "attachment_type": match.group("kind").lower(),
                "attachment_label": match.group("label").upper(),
            }
            continue

        if current is None:
            out.append(unit)
            continue

        current["text"] = f"{current['text'].rstrip()}\n\n{unit.get('text', '').strip()}".strip()
        current["page_numbers"] = sorted(set([
            *current.get("page_numbers", []),
            *unit.get("page_numbers", []),
        ]))

    flush_current()
    return out


# ---------- Semantic unit builder ----------

def split_appendix_into_blocks(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Parser experiment: appendix block detection is tuned to CUAD keyword and
    # trademark appendices. Disable if appendices are rare or paragraph chunks
    # work well enough.
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


def build_text_with_sic_annotations(text: str, typo_matches: List[Dict[str, Any]]) -> str:
    if not typo_matches:
        return text

    annotated_parts = []
    cursor = 0

    for typo in sorted(typo_matches, key=lambda item: item["start"]):
        start = typo["start"]
        end = typo["end"]
        if start < cursor:
            continue

        annotated_parts.append(text[cursor:end])
        annotated_parts.append(" [sic]")
        cursor = end

    annotated_parts.append(text[cursor:])
    return "".join(annotated_parts)


def annotate_suspected_typos(semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    typo_pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in COMMON_TYPO_CORRECTIONS) + r")\b",
        re.IGNORECASE,
    )

    for unit in semantic_units:
        text = unit.get("text", "")
        suspected_typos = []

        for match in typo_pattern.finditer(text):
            original = match.group(0)
            suggestion = COMMON_TYPO_CORRECTIONS[original.upper()]
            if original.islower():
                suggestion = suggestion.lower()
            elif original.istitle():
                suggestion = suggestion.title()

            suspected_typos.append({
                "text": original,
                "suggestion": suggestion,
                "start": match.start(),
                "end": match.end(),
                "annotation": f"{original} [sic]",
            })

        if suspected_typos:
            unit["suspected_typos"] = suspected_typos
            unit["text_with_sic_annotations"] = build_text_with_sic_annotations(text, suspected_typos)

    return semantic_units


def build_semantic_units(markdown_text: str, source_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pages = split_pages(markdown_text)
    base_metadata = detect_metadata_from_pages(pages)

    semantic_units: List[Dict[str, Any]]

    full_text = "\n\n".join(page for _, page in pages)
    use_article_section_parser = should_use_article_section_parser(full_text)

    # Parser experiment disabled: most specialized legal-format parsers are
    # bypassed so current PyMuPDF4LLM output is chunked by natural paragraphs.
    # Article/Section hierarchy is re-enabled only when clearly present because
    # sections are subordinate to articles in those contracts.
    # numeric_score = count_numeric_section_markers(full_text)
    # roman_score = count_roman_paragraph_markers(full_text)

    if use_article_section_parser:
        semantic_units = split_article_section_units(pages)
    # elif roman_score >= 2:
    #     top_units = split_roman_paragraph_units(pages)
    #     semantic_units = []
    #     for unit in top_units:
    #         semantic_units.extend(split_roman_paragraph_unit_into_subunits(unit))
    # elif numeric_score >= 2:
    #     top_units = split_numeric_top_sections(pages)
    #     semantic_units = []
    #     for unit in top_units:
    #         if unit["section_type"] == "appendix":
    #             semantic_units.extend(split_appendix_into_blocks(unit))
    #         else:
    #             semantic_units.extend(split_numeric_section_into_subsections(unit))
    # else:
    #     semantic_units = split_paragraph_fallback_units(pages)
    else:
        semantic_units = split_paragraph_fallback_units(pages)
    semantic_units, front_matter_metadata = compact_front_matter_units(semantic_units)
    semantic_units = merge_preamble_units(semantic_units)
    semantic_units = merge_section_continuation_units(semantic_units)
    semantic_units = split_restricted_keyword_catalog_units(semantic_units)
    semantic_units = merge_section_continuation_units(semantic_units)
    semantic_units = merge_attachment_preface_headings(semantic_units)
    semantic_units = merge_appendix_units_by_heading(semantic_units)
    semantic_units = split_appendix_units_by_category(semantic_units)
    semantic_units = merge_related_appendix_units(semantic_units)
    semantic_units = merge_attachment_units(semantic_units)
    semantic_units = annotate_suspected_typos(semantic_units)

    for i, unit in enumerate(semantic_units, start=1):
        stem = Path(source_file).stem
        unit["semantic_unit_id"] = f"{stem}|unit|{i}"
        unit["source"] = source_file
        unit["semantic_unit_token_count"] = count_tokens(unit["text"])

    return semantic_units, {**base_metadata, **front_matter_metadata}


# ---------- Final child-chunk builder ----------

def split_paragraph_by_tokens(paragraph: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    token_ids = get_tokenizer().encode(paragraph, add_special_tokens=False)
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

    paragraphs = split_paragraphs_with_list_items(text)
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
                    "front_matter_text": base_metadata.get("front_matter_text"),
                    "front_matter_pages": base_metadata.get("front_matter_pages"),
                    "section_type": unit.get("section_type"),
                    "article_number": unit.get("article_number"),
                    "article_title": unit.get("article_title"),
                    "section_number": unit.get("section_number"),
                    "section_title": unit.get("section_title"),
                    "attachment_type": unit.get("attachment_type"),
                    "attachment_label": unit.get("attachment_label"),
                    "contains_list_continuation": unit.get("contains_list_continuation"),
                    "continues_section": unit.get("continues_section"),
                    "continuation_label": unit.get("continuation_label"),
                    "suspected_typos": unit.get("suspected_typos"),
                    "text_with_sic_annotations": unit.get("text_with_sic_annotations"),
                    # Parser experiment disabled: these fields are only useful
                    # when specialized legal-section parsers are active.
                    # "roman_section_number": unit.get("roman_section_number"),
                    # "roman_section_title": unit.get("roman_section_title"),
                    # "subsection_number": unit.get("subsection_number"),
                    # "subsection_title": unit.get("subsection_title"),
                    # "subunit_label": unit.get("subunit_label"),
                    # "subunit_title": unit.get("subunit_title"),
                    # "roman_subunit_label": unit.get("roman_subunit_label"),
                    # "roman_subunit_title": unit.get("roman_subunit_title"),
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

    markdown_text = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        footer=False,
        header=True,
        page_separators=True
    )

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
            # print(f"Document Title: {document_metadata.get('document_title')}")
            # print(f"Company Names: {document_metadata.get('company_names')}")
            # print(f"Party Count: {document_metadata.get('party_count')}")
            # print(f"Source: {unit['source']}")
            # print(f"Semantic Unit ID: {unit['semantic_unit_id']}")
            # print(f"Section Type: {unit.get('section_type')}")
            print(f"Page Numbers: {unit.get('page_numbers')}")
            # Parser experiment disabled: specialized parser metadata is not
            # populated in paragraph-fallback mode.
            # print(f"Article: {unit.get('article_number')} - {unit.get('article_title')}")
            # print(f"Section: {unit.get('section_number')} - {unit.get('section_title')}")
            # print(f"Roman Section: {unit.get('roman_section_number')} - {unit.get('roman_section_title')}")
            # print(f"Subsection: {unit.get('subsection_number')} - {unit.get('subsection_title')}")
            # print(f"Roman Subunit: {unit.get('roman_subunit_label')} - {unit.get('roman_subunit_title')}")
            # print(f"Subunit: {unit.get('subunit_label')} - {unit.get('subunit_title')}")
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


def load_pdf_alternative_method(
    relative_path: str,
) -> List[Document]:
    root_path = Path(__file__).resolve().parents[3]
    pdf_path = root_path / Path(relative_path)
    assert pdf_path.exists(), f"File not found: {pdf_path}"

    doc = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        footer=False,
        header=True,
        page_separators=True
    )
    print(doc)


if __name__ == "__main__":
    # relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
    relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf"
    # relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf"
    # relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement.pdf"
    # relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement.pdf"
    load_pdf(relative_path, debug=True)

    # python src/contract_copilot/indexer/ocr_loader.py
