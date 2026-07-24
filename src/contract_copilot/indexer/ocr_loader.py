"""Extract PDFs and turn legal section hierarchies into embedding-sized chunks.

The pipeline is intentionally structure-first: it finds primary legal sections,
recursively descends into subsections and clauses only when a parent is too large,
and uses overlapping token windows only when no finer legal structure is available.
Only the resulting leaves are indexed; their heading paths retain the context of
the unsaved parent sections.
"""

import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_DIR = Path("data/raw/CUAD_v1/full_contract_pdf")

ARTICLE_RE = re.compile(
    r"^ARTICLE\s+(?P<label>[IVXLCDM]+|\d+)(?:(?:\s*[-:.]\s*|\s+)(?P<title>.+))?$",
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    r"^SECTION\s+(?P<label>\d+(?:\.\d+)*)(?:(?:\s*[-:.]\s*|\s+)(?P<title>.+))?$",
    re.IGNORECASE,
)
SECTION_SIGN_RE = re.compile(
    r"^§\s*(?P<label>\d+(?:\.\d+)*)(?:\s+(?P<title>.+))?$",
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(
    r"^(?P<label>\d{1,2}(?:\.\d+)*)(?:(?:\.)?\s+|\.(?=[A-Z]))(?P<title>.+)$"
)
ROMAN_RE = re.compile(
    r"^(?P<label>[IVXLCDM]{1,8})\.\s+(?P<title>.+)$",
    re.IGNORECASE,
)
ATTACHMENT_RE = re.compile(
    r"^(?P<kind>(?i:Appendix|Schedule|Exhibit|Annex|Attachment|Addendum))"
    r"\s+(?P<label>[A-Z0-9]+[A-Z0-9.-]*)(?:\s*[-:.]?\s*(?P<title>.*))?$",
)
PAREN_RE = re.compile(
    r"^\((?P<label>[a-z]|\d+|[ivxlcdm]+)\)\s+(?P<title>.+)$",
    re.IGNORECASE,
)
TOC_ROW_RE = re.compile(r"(?:\.{2,}|\s{2,})\s*\d+\s*$")
INLINE_ARTICLE_RE = re.compile(
    r"^(?P<body>.+?[.!?])\s+(?P<header>\*\*ARTICLE\s+(?:[IVXLCDM]+|\d+)"
    r"(?:\s+[A-Z][A-Z ;,&/-]+)?\*\*)\s*$",
    re.IGNORECASE,
)
LEADING_BOLD_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:\*\*|__)(?P<heading>.+?)(?:\*\*|__)"
    r"(?:\s+(?P<body>.*))?$"
)
BARE_ATTACHMENT_RE = re.compile(
    r"^(?P<kind>Appendix|Schedule|Exhibit|Annex|Attachment|Addendum)$",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'”’\)\]]+)?(?=\s+|$)")
LIST_START_RE = re.compile(r"(?m)^(?:[-+*•]\s+|\([a-zivxlcdm\d]+\)\s+)", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^(?:[-+*•]\s*|\|?\s*•\s*)(?P<body>.*?)(?:\|)?$")
ATTACHMENT_ENTRY_HEADING_RE = re.compile(
    r"^(?P<label>.+?\s+Restricted\s+(?:Trademark\s+Terms|Key\s+Words))$",
    re.IGNORECASE,
)
CROSS_REFERENCE_RE = re.compile(
    r"\b(?P<kind>Section|Article|Appendix|Exhibit|Schedule)"
    r"(?:\s+(?P<label>[A-Z0-9]+(?:\.\d+)*))?|"
    r"(?P<section_sign>§\s*\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
INTERFACE_CONTROLS = {"Accept.", "Agree.", "Submit.", "Cancel.", "Print."}
ABBREVIATIONS = {
    "cf.", "corp.", "dr.", "e.g.", "fig.", "i.e.", "inc.", "ltd.",
    "mr.", "mrs.", "no.", "sec.", "u.k.", "u.s.", "vs.",
}


# These small immutable records keep extraction, structural parsing, and output
# assembly separate. That makes recursive splitting easier to reason about than
# mutating LangChain Documents throughout the pipeline.
@dataclass(frozen=True)
class _Line:
    page: int
    text: str
    classified_header: bool = False
    toc_page: bool = False
    bbox: tuple[float, float, float, float] | None = None
    page_width: float | None = None


@dataclass(frozen=True)
class _Heading:
    label: str
    title: str
    depth: int
    kind: str
    remainder: str = ""
    trusted: bool = False
    consume_next: bool = False

    @property
    def display(self) -> str:
        return " ".join(part for part in (self.label, self.title) if part).strip()


@dataclass(frozen=True)
class _Block:
    path: tuple[str, ...]
    labels: tuple[str, ...]
    body: tuple[_Line, ...]
    depth: int
    kind: str
    split_strategy: str
    list_context: str | None = None
    layout: tuple[_Line, ...] = ()


@dataclass(frozen=True)
class _Leaf:
    text: str
    pages: tuple[int, ...]
    section_path: tuple[str, ...]
    leaf_label: str
    split_strategy: str
    list_context: str | None = None


@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBED_MODEL_ID)


def _encode(text: str) -> list[int]:
    return _get_tokenizer().encode(text, add_special_tokens=False)


def count_tokens(text: str) -> int:
    """Count tokens with the same tokenizer used to enforce chunk limits."""
    return len(_encode(text))


def _decode(token_ids: list[int]) -> str:
    return _get_tokenizer().decode(token_ids, skip_special_tokens=True).strip()


def _clean_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return re.sub(r"^(\*\*|__)(.+)\1$", r"\2", text).strip()


def _strip_markup(line: str) -> tuple[str, int]:
    match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    text = match.group(2).strip() if match else line.strip()
    text = text.strip("| ")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\*\*<u>(.*?)</u>\*\*", r"\1", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"(?:\*\*|__)(.*?)(?:\*\*|__)", r"\1", text).strip()
        text = re.sub(r"<u>(.*?)</u>", r"\1", text, flags=re.IGNORECASE).strip()
    return text, len(match.group(1)) if match else 0


def _split_title(text: str, trusted: bool) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "", ""

    match = re.match(r"(?P<title>.*?[.:])(?:\s+(?P<body>.+))?$", text)
    if match and len(match.group("title").split()) <= 20:
        return match.group("title").strip(" .:"), (match.group("body") or "").strip()

    if trusted or len(text.split()) <= 20:
        return text.strip(" .:"), ""
    return "", text


def _is_toc_row(text: str) -> bool:
    if TOC_ROW_RE.search(text):
        return True
    # Layout extraction often collapses a TOC row to "ARTICLE 2 Licenses 13".
    return bool(
        re.match(r"^(?:ARTICLE|SECTION)\s+\S+\s+.+\s+\d+$", text, re.IGNORECASE)
    )


def _parse_heading(line: _Line) -> _Heading | None:
    """Parse one line as a supported legal heading, without accepting it yet.

    Layout-classified and Markdown headings are marked as trusted. Regex-only
    candidates are validated later against sibling headings, which prevents an
    isolated citation or numbered sentence from becoming a section boundary.
    """
    text, markdown_depth = _strip_markup(line.text)
    styled_body = ""
    styled = LEADING_BOLD_RE.match(line.text.strip().strip("| "))
    if styled:
        # PyMuPDF4LLM commonly bolds only the legal heading and leaves the
        # clause body on the same line. Preserve that boundary before stripping
        # Markdown, otherwise the complete line looks like an oversized title.
        text, _ = _strip_markup(styled.group("heading"))
        styled_body = (styled.group("body") or "").strip()
    trusted = line.classified_header or markdown_depth > 0 or styled is not None
    if not text or line.toc_page or _is_toc_row(line.text):
        return None

    bare_attachment = BARE_ATTACHMENT_RE.match(text)
    if bare_attachment and trusted:
        return _Heading(
            bare_attachment.group("kind").title(),
            "",
            1,
            "attachment",
            styled_body,
            trusted,
        )

    match = ATTACHMENT_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title") or "", trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        label = f"{match.group('kind').title()} {match.group('label').upper()}"
        exact_or_styled = trusted or (
            len(text.split()) <= 12 and (not title or title[:1].isupper())
        )
        if exact_or_styled:
            return _Heading(label, title, 1, "attachment", remainder, trusted)

    match = ARTICLE_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title") or "", trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        if title or not match.group("title"):
            return _Heading(
                f"ARTICLE {match.group('label').upper()}",
                title,
                1,
                "article",
                remainder,
                trusted,
            )

    match = SECTION_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title") or "", trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        if title or not match.group("title"):
            number = match.group("label")
            return _Heading(
                f"Section {number}",
                title,
                number.count(".") + 1,
                "section",
                remainder,
                trusted,
            )

    match = SECTION_SIGN_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title") or "", trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        if title or not match.group("title"):
            return _Heading(
                f"§ {match.group('label')}",
                title,
                2,
                "section_sign",
                remainder,
                trusted,
            )

    match = NUMBERED_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title"), trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        if title and (trusted or title[:1].isupper()):
            number = match.group("label")
            suffix = "." if "." not in number else ""
            return _Heading(
                f"{number}{suffix}",
                title,
                number.count(".") + 1,
                "numeric",
                remainder,
                trusted,
            )

    match = ROMAN_RE.match(text)
    if match:
        title, remainder = _split_title(match.group("title"), trusted)
        remainder = " ".join(part for part in (remainder, styled_body) if part)
        if title and (trusted or title[:1].isupper()):
            return _Heading(
                f"{match.group('label').upper()}.",
                title,
                1,
                "roman",
                remainder,
                trusted,
            )

    if trusted and len(text.split()) <= 15 and not text.endswith((".", ";", "?", "!")):
        return _Heading(text, "", markdown_depth or 1, "layout", trusted=True)

    return None


def _page_lines(page_chunks: list[dict[str, Any]]) -> list[_Line]:
    """Flatten PyMuPDF4LLM pages while preserving page and header provenance."""
    lines: list[_Line] = []
    for index, chunk in enumerate(page_chunks, start=1):
        metadata = chunk.get("metadata") or {}
        page_number = int(metadata.get("page_number") or index)
        text = unicodedata.normalize("NFKC", chunk.get("text") or "")
        toc_page = bool(re.search(r"\bTABLE\s+OF\s+CONTENTS\b", text, re.IGNORECASE))
        header_ranges = [
            tuple(box.get("pos", (0, 0)))
            for box in chunk.get("page_boxes", [])
            if box.get("class") == "section-header"
        ]

        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            start, end = cursor, cursor + len(raw_line)
            cursor = end
            cleaned = _clean_line(raw_line)
            if cleaned in INTERFACE_CONTROLS:
                continue
            classified = any(start < stop and end > begin for begin, stop in header_ranges)
            inline_article = INLINE_ARTICLE_RE.match(cleaned)
            if inline_article:
                # Some PDFs glue a bold ARTICLE heading to the previous sentence.
                # Separate it here so the structural parser sees a real boundary.
                lines.append(_Line(page_number, inline_article.group("body"), False, toc_page))
                lines.append(_Line(page_number, inline_article.group("header"), classified, toc_page))
            else:
                lines.append(_Line(page_number, cleaned, classified, toc_page))

    return lines


def _layout_lines(page_chunks: list[dict[str, Any]]) -> list[_Line]:
    """Return visual PDF lines added by ``load_pdf`` for attachment parsing."""
    lines = []
    for index, chunk in enumerate(page_chunks, start=1):
        page = int((chunk.get("metadata") or {}).get("page_number") or index)
        for item in chunk.get("layout_lines", ()):
            text = _clean_line(item.get("text") or "")
            if text and text not in INTERFACE_CONTROLS:
                lines.append(
                    _Line(
                        page,
                        text,
                        bbox=tuple(item["bbox"]),
                        page_width=float(item["page_width"]),
                    )
                )
    return lines


def _interface_artifacts(page_chunks: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Collect only confirmed standalone controls for optional debug reporting."""
    artifacts = []
    for index, chunk in enumerate(page_chunks, start=1):
        page = int((chunk.get("metadata") or {}).get("page_number") or index)
        for raw_line in (chunk.get("text") or "").splitlines():
            text = _clean_line(raw_line)
            if text in INTERFACE_CONTROLS:
                artifacts.append((page, text))
    return artifacts


def _heading_group_key(heading: _Heading) -> tuple[str, int, str]:
    """Return the sibling family used to validate regex-derived headings."""
    label = heading.label.rstrip(".")
    parent = ""
    number = re.search(r"\d+(?:\.\d+)*", label)
    if number and "." in number.group(0):
        parent = number.group(0).rsplit(".", 1)[0]
    return heading.kind, heading.depth, parent


def _validated_headings(lines: tuple[_Line, ...] | list[_Line]) -> list[tuple[int, _Heading]]:
    """Find headings and reject weak candidates that lack structural support.

    A candidate survives when layout/Markdown marked it as a header or when at
    least two distinct labels occur at the same kind, depth, and numeric parent.
    Attachments are allowed as standalone boundaries because a contract may have
    only one exhibit or schedule.
    """
    candidates: list[tuple[int, _Heading]] = []
    consumed_indices: set[int] = set()
    for index, line in enumerate(lines):
        if index in consumed_indices:
            continue
        heading = _parse_heading(line)
        if heading is None:
            continue

        if (
            heading.kind == "article"
            and not heading.title
            and index + 1 < len(lines)
        ):
            next_line = lines[index + 1]
            next_text, _ = _strip_markup(next_line.text)
            if (
                next_line.page == line.page
                and next_text
                and len(next_text.split()) <= 12
                and next_text.isupper()
                and (next_line.classified_header or line.classified_header)
            ):
                # Layout extraction may emit "ARTICLE I" and its all-caps title as
                # adjacent boxes. Treat them as one heading, not two sections.
                heading = replace(heading, title=next_text.title(), consume_next=True)
                consumed_indices.add(index + 1)

        candidates.append((index, heading))

    grouped: dict[tuple[str, int, str], list[_Heading]] = {}
    for _, heading in candidates:
        grouped.setdefault(_heading_group_key(heading), []).append(heading)

    accepted = []
    for index, heading in candidates:
        siblings = grouped[_heading_group_key(heading)]
        distinct_labels = {sibling.label.lower() for sibling in siblings}
        # A lone Roman-looking label is commonly ordinary prose. Requiring I or
        # II anchors the sequence while still supporting later Roman siblings.
        if heading.kind == "roman" and not distinct_labels.intersection({"i.", "ii."}):
            continue
        if (
            heading.trusted
            or len(distinct_labels) >= 2
            or (heading.kind == "attachment" and len(heading.display.split()) <= 12)
        ):
            accepted.append((index, heading))
    return accepted


def _path_prefix(path: tuple[str, ...], max_tokens: int | None = None) -> str:
    """Render a compact Markdown heading path that fits inside the chunk budget."""
    selected = path
    prefix = ""
    while selected:
        prefix = "\n".join(
            f"{'#' * min(index + 2, 6)} {heading}"
            for index, heading in enumerate(selected)
        )
        if max_tokens is None or count_tokens(prefix) < max_tokens:
            return prefix
        if len(selected) == 1:
            # Pathological OCR titles can exceed the entire budget. Retain a
            # shortened leaf title so some legal context still reaches embedding.
            return _decode(_encode(prefix)[: max(1, max_tokens // 2)])
        selected = selected[1:]
    return prefix


def _body_text(lines: tuple[_Line, ...]) -> str:
    text = "\n".join(line.text for line in lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _render(path: tuple[str, ...], body: str, max_tokens: int | None = None) -> str:
    return "\n\n".join(
        part for part in (_path_prefix(path, max_tokens), body.strip()) if part
    ).strip()


def _make_block(
    heading: _Heading,
    heading_line: _Line,
    lines: list[_Line] | tuple[_Line, ...],
    parent: _Block | None = None,
) -> _Block:
    """Create a structural block and inherit its parent's heading path."""
    # Keep a zero-text page marker so header-only sections retain provenance.
    body = [replace(heading_line, text="", classified_header=False), *lines]
    if heading.remainder:
        body.insert(1, replace(heading_line, text=heading.remainder, classified_header=False))
    path = (*parent.path, heading.display) if parent else (heading.display,)
    labels = (*parent.labels, heading.label) if parent else (heading.label,)
    strategy = "nested_clause" if heading.kind == "clause" else (
        "subsection" if parent else "section"
    )
    return _Block(path, labels, tuple(body), heading.depth, heading.kind, strategy)


def _attachment_layout_slice(
    layout_lines: list[_Line],
    heading: _Heading,
    pages: set[int],
    next_heading: _Heading | None = None,
) -> tuple[_Line, ...]:
    """Keep visual lines belonging to one attachment, beginning after its title."""
    candidates = [line for line in layout_lines if line.page in pages]
    heading_text = heading.display.casefold()
    for index, line in enumerate(candidates):
        if line.text.casefold() == heading_text or line.text.casefold() == heading.label.casefold():
            attachment_lines = candidates[index + 1:]
            if next_heading:
                next_labels = {
                    next_heading.display.casefold(),
                    next_heading.label.casefold(),
                }
                attachment_lines = attachment_lines[:next(
                    (
                        offset for offset, candidate in enumerate(attachment_lines)
                        if candidate.text.casefold() in next_labels
                    ),
                    len(attachment_lines),
                )]
            return tuple(attachment_lines)
    return ()


def _build_primary_sections(page_chunks: list[dict[str, Any]]) -> list[_Block]:
    """Partition a document into complete top-level legal sections.

    Explicit legal markers take precedence over generic layout headings. The
    shallowest repeated legal depth becomes the document root, while attachments
    always begin independent primary units. Text before the first boundary is
    retained as a preamble rather than discarded.
    """
    lines = _page_lines(page_chunks)
    layout_lines = _layout_lines(page_chunks)
    headings = _validated_headings(lines)
    if not headings:
        return [_Block(("Document",), ("Document",), tuple(lines), 1, "document", "section")]

    legal_headings = [item for item in headings if item[1].kind != "layout"]
    # Generic layout titles are useful only when no recognizable legal hierarchy
    # exists; mixing both would fragment sections at decorative headings.
    boundary_pool = legal_headings or headings
    root_depth = min(
        (heading.depth for _, heading in boundary_pool if heading.kind != "attachment"),
        default=1,
    )
    boundaries = [
        (index, heading)
        for index, heading in boundary_pool
        if heading.depth == root_depth or heading.kind == "attachment"
    ]
    sections: list[_Block] = []

    first_index = boundaries[0][0]
    if any(line.text for line in lines[:first_index]):
        sections.append(
            _Block(
                ("Preamble",),
                ("Preamble",),
                tuple(lines[:first_index]),
                root_depth,
                "preamble",
                "section",
            )
        )

    for boundary_index, (start, heading) in enumerate(boundaries):
        end = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else len(lines)
        content_start = start + 1 + int(heading.consume_next)
        section = _make_block(heading, lines[start], lines[content_start:end])
        if heading.kind == "attachment" and layout_lines:
            pages = {line.page for line in section.body}
            next_heading = (
                boundaries[boundary_index + 1][1]
                if boundary_index + 1 < len(boundaries)
                else None
            )
            section = replace(
                section,
                layout=_attachment_layout_slice(layout_lines, heading, pages, next_heading),
            )
        sections.append(section)
    return sections


def _parse_parenthetical_headings(lines: tuple[_Line, ...], depth: int) -> list[tuple[int, _Heading]]:
    """Infer one consistent `(a)`, `(1)`, or `(i)` sibling family."""
    raw: list[tuple[int, re.Match[str], bool]] = []
    for index, line in enumerate(lines):
        text, markdown_depth = _strip_markup(line.text)
        match = PAREN_RE.match(text)
        if match:
            raw.append((index, match, line.classified_header or markdown_depth > 0))
    if len(raw) < 2:
        return []

    labels = [match.group("label").lower() for _, match, _ in raw]
    # Single-letter Roman numerals overlap with alphabetic clauses. Multi-character
    # labels establish a Roman sequence; otherwise letters are the safer reading.
    if all(label.isdigit() for label in labels):
        family = "number"
    elif any(len(label) > 1 for label in labels):
        family = "roman"
    else:
        family = "letter"

    out = []
    for index, match, trusted in raw:
        label = match.group("label").lower()
        if family == "number" and not label.isdigit():
            continue
        if family == "roman" and not re.fullmatch(r"[ivxlcdm]+", label):
            continue
        # Parenthetical clauses usually begin directly with prose. Only keep a
        # genuinely short caption in the path; otherwise retain all prose as body.
        title, remainder = _split_title(match.group("title"), trusted=False)
        out.append(
            (
                index,
                _Heading(f"({label})", title, depth + 1, "clause", remainder, trusted),
            )
        )
    return out if len({heading.label for _, heading in out}) >= 2 else []


def _nested_boundaries(block: _Block) -> list[tuple[int, _Heading]]:
    """Return only the next structural level below a block."""
    structural = [
        (index, heading)
        for index, heading in _validated_headings(block.body)
        if heading.depth > block.depth and heading.kind not in {"attachment", "layout"}
    ]
    if structural:
        # Skipping directly to a deeper level would lose intermediate parent
        # context and could mix grandchildren from different subsections.
        next_depth = min(heading.depth for _, heading in structural)
        return [(index, heading) for index, heading in structural if heading.depth == next_depth]
    return _parse_parenthetical_headings(block.body, block.depth)


def _split_block_structurally(block: _Block) -> list[_Block]:
    """Split a block into its immediate children without losing leading prose."""
    boundaries = _nested_boundaries(block)
    if not boundaries:
        return []

    children: list[_Block] = []
    first_index = boundaries[0][0]
    if any(line.text for line in block.body[:first_index]):
        # Definitions and lead-in language before the first child still belong to
        # the parent. Preserve them as a structural leaf with the parent's path.
        children.append(replace(block, body=block.body[:first_index]))

    for boundary_index, (start, heading) in enumerate(boundaries):
        end = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else len(block.body)
        content_start = start + 1 + int(heading.consume_next)
        children.append(_make_block(heading, block.body[start], block.body[content_start:end], parent=block))
    return children


def _expanded_list_lines(lines: tuple[_Line, ...]) -> list[tuple[_Line, bool]]:
    """Split OCR lines containing several bullet glyphs into source list items."""
    expanded = []
    for line in lines:
        text = line.text.replace("<br>", " ").strip("| ")
        if not text or text == "---":
            expanded.append((replace(line, text=text), False))
            continue
        markers = list(re.finditer(r"(?:^[-+*]​?\s+|(?:^|(?<=\s))•\s*)", text))
        if not markers:
            expanded.append((replace(line, text=text), False))
            continue
        prefix = text[:markers[0].start()].strip()
        if prefix:
            expanded.append((replace(line, text=prefix), False))
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            item = text[marker.end():end].strip(" |")
            if item:
                expanded.append((replace(line, text=f"- {item}"), True))
    return expanded


def _list_item_body(text: str) -> str:
    match = LIST_ITEM_RE.match(text.strip())
    return (match.group("body") if match else text).strip()


def _is_complete_obligation(text: str) -> bool:
    """Keep sentence-like duties independent; terse examples may be packed."""
    return bool(re.search(r"[.!?](?:\s*\([^)]*\))?\s*$", text))


def _list_block(block: _Block, lines: list[_Line], context: str | None) -> _Block:
    path = (*block.path, context) if context and context not in block.path else block.path
    labels = (*block.labels, context) if context and context not in block.labels else block.labels
    return _Block(
        path,
        labels,
        tuple(lines),
        block.depth + int(bool(context)),
        "list",
        "list",
        context,
    )


def _split_list_blocks(block: _Block, max_tokens: int) -> list[_Block]:
    """Split list runs while repeating only their exact source lead-in."""
    expanded = [item for item in _expanded_list_lines(block.body) if item[0].text]
    if sum(is_item for _, is_item in expanded) < 2:
        return []

    children: list[_Block] = []
    prose: list[_Line] = []
    packed: list[_Line] = []
    context: str | None = None

    def flush_prose() -> None:
        nonlocal prose
        if any(line.text for line in prose):
            children.append(replace(block, body=tuple(prose), layout=()))
        prose = []

    def flush_packed() -> None:
        nonlocal packed
        if packed:
            children.append(_list_block(block, packed, context))
        packed = []

    for index, (line, is_item) in enumerate(expanded):
        if not is_item:
            flush_packed()
            prose.append(line)
            continue

        if prose:
            nearest = next((item.text for item in reversed(prose) if item.text), "")
            context = nearest if nearest.endswith(":") else None
            flush_prose()

        item = _list_item_body(line.text)
        if item.endswith(":"):
            flush_packed()
            context = item
            continue

        if _is_complete_obligation(item):
            flush_packed()
            children.append(_list_block(block, [line], context))
            continue

        candidate = [*packed, line]
        candidate_block = _list_block(block, candidate, context)
        if packed and count_tokens(_render(candidate_block.path, _body_text(candidate_block.body))) > max_tokens:
            flush_packed()
        packed.append(line)

        next_is_item = index + 1 < len(expanded) and expanded[index + 1][1]
        if not next_is_item:
            flush_packed()

    flush_packed()
    flush_prose()
    return children


def _attachment_entry_heading(text: str) -> str | None:
    match = ATTACHMENT_ENTRY_HEADING_RE.match(text.strip())
    if not match or match.group("label").casefold().startswith("partner restricted"):
        return None
    return match.group("label").strip()


def _infer_attachment_label(text: str) -> str | None:
    """Infer a row label only from repetition, URLs, or an explicit ``See``."""
    see = re.match(r"(?P<label>.+?)\s+See\s+", text, re.IGNORECASE)
    if see:
        return see.group("label").strip(" ,:;- ")

    first = re.match(r"(?P<label>[A-Z][A-Za-z0-9&'/.-]*)\b", text)
    if first and re.search(
        rf"\b{re.escape(first.group('label'))}\b",
        text[first.end():],
        re.IGNORECASE,
    ):
        return first.group("label")

    url = re.search(r"\b(?:https?://|www\.)", text, re.IGNORECASE)
    if url:
        prefix = text[:url.start()].strip(" ,:;- ")
        if prefix and len(prefix.split()) <= 4:
            return prefix
    return None


def _split_attachment_entries(block: _Block) -> list[_Block]:
    """Recover appendix rows from visual PDF lines without guessing brands."""
    if block.kind != "attachment" or not block.layout:
        return []

    entries: list[tuple[str, list[_Line]]] = []
    current_label: str | None = None
    current_lines: list[_Line] = []
    fallback_index = 0

    def flush() -> None:
        nonlocal current_label, current_lines
        if current_label and current_lines:
            entries.append((current_label, current_lines))
        current_label, current_lines = None, []

    previous: _Line | None = None
    for line in block.layout:
        text = line.text.strip()
        if text.casefold() in {
            "list of restricted trademark terms",
            "partner restricted trademark terms",
        }:
            continue

        explicit = _attachment_entry_heading(text)
        if explicit:
            flush()
            current_label = explicit
            previous = line
            continue
        if current_label and " restricted " in current_label.casefold():
            current_lines.append(line)
            previous = line
            continue

        label = _infer_attachment_label(text)
        if label:
            flush()
            current_label, current_lines = label, [line]
        else:
            usable_width = (
                previous.page_width - 2 * previous.bbox[0]
                if previous and previous.page_width and previous.bbox
                else 0
            )
            wide = bool(
                previous
                and previous.bbox
                and usable_width > 0
                and (previous.bbox[2] - previous.bbox[0]) / usable_width >= 0.82
            )
            continues = bool(
                current_lines
                and previous
                and (
                    previous.text.rstrip().endswith(",")
                    or wide
                    or text.casefold().startswith("in addition,")
                )
            )
            if continues:
                current_lines.append(line)
            else:
                flush()
                fallback_index += 1
                current_label = f"Appendix entry {fallback_index}"
                current_lines = [line]
        previous = line
    flush()

    return [
        _Block(
            (*block.path, label),
            (*block.labels, label),
            tuple(lines),
            block.depth + 1,
            "attachment_entry",
            "attachment_entry",
        )
        for label, lines in entries
    ]


def _pages_for_offsets(
    lines: tuple[_Line, ...],
    start_offset: int,
    end_offset: int,
) -> tuple[int, ...]:
    """Map tokenizer character offsets back to the source pages they overlap."""
    pages = []
    cursor = 0
    for line in lines:
        line_end = cursor + len(line.text)
        if cursor < end_offset and line_end >= start_offset and line.text:
            pages.append(line.page)
        cursor = line_end + 1
    return tuple(sorted(set(pages)))


def _sentence_units(text: str) -> list[tuple[int, int]]:
    """Return conservative paragraph, list-item, and sentence spans."""
    boundaries = {0, len(text)}
    for match in re.finditer(r"\n\s*\n", text):
        boundaries.update((match.start(), match.end()))
    boundaries.update(match.start() for match in LIST_START_RE.finditer(text))

    for match in SENTENCE_END_RE.finditer(text):
        before = text[:match.end()].rstrip()
        last_word = before.split()[-1].strip("\"'“”‘’()[]").lower()
        if last_word in ABBREVIATIONS or re.search(r"(?:\b[A-Z]\.){2,}$", before):
            continue
        after = text[match.end():].lstrip()
        if after and after[0].islower():
            continue
        boundaries.add(match.end())

    points = sorted(boundaries)
    spans = []
    for start, end in zip(points, points[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _hard_token_windows(
    block: _Block,
    prefix: str,
    body: str,
    start_offset: int,
    end_offset: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[_Leaf]:
    """Use fixed token windows for one unit that cannot otherwise fit."""
    prefix_tokens = count_tokens(f"{prefix}\n\n")
    budget = max_tokens - prefix_tokens
    if budget <= 0:
        raise ValueError("Section heading path exceeds max_tokens")

    tokenizer = _get_tokenizer()
    unit_text = body[start_offset:end_offset]
    encoded = tokenizer(unit_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = list(encoded.get("offset_mapping") or [])
    overlap = min(overlap_tokens, max(0, budget - 1))
    leaves: list[_Leaf] = []
    start = 0
    while start < len(token_ids):
        end = min(start + budget, len(token_ids))
        chunk_body = _decode(token_ids[start:end])
        chunk_text = "\n\n".join(part for part in (prefix, chunk_body) if part)
        # Decoding and re-encoding can change token counts for some tokenizers.
        # Tighten the window until the final indexed string satisfies the limit.
        while count_tokens(chunk_text) > max_tokens and end > start + 1:
            end -= 1
            chunk_body = _decode(token_ids[start:end])
            chunk_text = "\n\n".join(part for part in (prefix, chunk_body) if part)

        if offsets and start < len(offsets) and end - 1 < len(offsets):
            pages = _pages_for_offsets(
                block.body,
                start_offset + offsets[start][0],
                start_offset + offsets[end - 1][1],
            )
        else:
            pages = tuple(sorted({line.page for line in block.body}))
        leaves.append(
            _Leaf(
                chunk_text,
                pages,
                block.path,
                block.labels[-1],
                "token_fallback",
                block.list_context,
            )
        )
        if end == len(token_ids):
            break
        start = end - overlap
    return leaves


def _token_fallback(block: _Block, max_tokens: int, overlap_tokens: int) -> list[_Leaf]:
    """Pack complete text units, splitting tokens only inside an oversized unit."""
    prefix = _path_prefix(block.path, max_tokens)
    body = _body_text(block.body)
    units = _sentence_units(body)
    leaves: list[_Leaf] = []
    unit_index = 0

    while unit_index < len(units):
        chunk_start = unit_index
        chunk_end = chunk_start
        chunk_body = ""
        while chunk_end < len(units):
            candidate = body[units[chunk_start][0]:units[chunk_end][1]].strip()
            candidate_text = "\n\n".join(part for part in (prefix, candidate) if part)
            if count_tokens(candidate_text) > max_tokens:
                break
            chunk_body = candidate
            chunk_end += 1

        if chunk_end == chunk_start:
            start_offset, end_offset = units[chunk_start]
            leaves.extend(
                _hard_token_windows(
                    block,
                    prefix,
                    body,
                    start_offset,
                    end_offset,
                    max_tokens,
                    overlap_tokens,
                )
            )
            unit_index += 1
            continue

        start_offset = units[chunk_start][0]
        end_offset = units[chunk_end - 1][1]
        pages = _pages_for_offsets(block.body, start_offset, end_offset)
        leaves.append(
            _Leaf(
                "\n\n".join(part for part in (prefix, chunk_body) if part),
                pages,
                block.path,
                block.labels[-1],
                "token_fallback",
                block.list_context,
            )
        )
        if chunk_end == len(units):
            break

        # Carry only complete trailing units into the next chunk. If even the
        # last unit is larger than the requested overlap, start without overlap.
        overlap_start = chunk_end
        while overlap_start > chunk_start:
            candidate_start = overlap_start - 1
            overlap_text = body[units[candidate_start][0]:units[chunk_end - 1][1]]
            if count_tokens(overlap_text) > overlap_tokens:
                break
            overlap_start = candidate_start
        unit_index = overlap_start if overlap_start > chunk_start else chunk_end

    return leaves


def _leaf_chunks(block: _Block, max_tokens: int, overlap_tokens: int) -> list[_Leaf]:
    """Recursively descend through structure, then fall back to token windows."""
    # Lists and appendix rows are already complete source units, so preserve
    # them even when their combined parent happens to fit the token budget.
    # Numbered legal children remain size-driven below.
    source_children = _split_attachment_entries(block)
    if not source_children and block.kind != "list" and not _nested_boundaries(block):
        source_children = _split_list_blocks(block, max_tokens)
    if source_children:
        leaves = []
        for child in source_children:
            leaves.extend(_leaf_chunks(child, max_tokens, overlap_tokens))
        return leaves

    rendered = _render(block.path, _body_text(block.body), max_tokens)
    if count_tokens(rendered) <= max_tokens:
        pages = tuple(sorted({line.page for line in block.body}))
        return [
            _Leaf(
                rendered,
                pages,
                block.path,
                block.labels[-1],
                block.split_strategy,
                block.list_context,
            )
        ]

    children = _split_block_structurally(block)
    if children:
        # Structurally complete siblings do not overlap. Overlap is reserved for
        # the final token fallback where there is no meaningful legal boundary.
        leaves = []
        for child in children:
            leaves.extend(_leaf_chunks(child, max_tokens, overlap_tokens))
        return leaves
    return _token_fallback(block, max_tokens, overlap_tokens)


def _path_metadata(source_file: str) -> dict[str, Any]:
    """Derive stable CUAD metadata from a source path when available."""
    path = Path(source_file)
    parts = path.parts
    try:
        corpus_index = parts.index("full_contract_pdf")
    except ValueError:
        corpus_index = -1
    document_id = path.stem
    return {
        "document_id": document_id,
        "source_path": source_file,
        "file_name": path.name,
        "dataset": next((part for part in parts if part.endswith("_v1")), None),
        "corpus": "full_contract_pdf" if corpus_index >= 0 else None,
        "part": parts[corpus_index + 1] if corpus_index >= 0 and len(parts) > corpus_index + 1 else None,
        "contract_type": parts[corpus_index + 2] if corpus_index >= 0 and len(parts) > corpus_index + 2 else None,
        "document_title": re.sub(r"[_-]+", " ", document_id).strip(),
    }


def _reference_targets(sections: list[_Block], semantic_ids: list[str]) -> dict[str, str]:
    """Map common legal citation spellings to their primary section IDs."""
    targets = {}
    for section, semantic_id in zip(sections, semantic_ids):
        label = section.labels[0].rstrip(".")
        display = section.path[0]
        targets[label.casefold()] = semantic_id
        targets[display.casefold()] = semantic_id
        number = re.match(r"(?P<number>\d+(?:\.\d+)*)\.?\b", label)
        if number:
            targets[f"section {number.group('number')}".casefold()] = semantic_id
        if label.casefold().startswith("article "):
            targets[label.casefold()] = semantic_id
        if section.kind == "attachment":
            targets[label.casefold()] = semantic_id
    return targets


def _cross_references(text: str, targets: dict[str, str]) -> list[dict[str, str | None]]:
    """Extract explicit legal references and retain unresolved citations."""
    body = re.sub(r"(?m)^#{1,6}\s+.*(?:\n|$)", "", text)
    references = []
    seen = set()
    for match in CROSS_REFERENCE_RE.finditer(body):
        if match.group("section_sign"):
            canonical = re.sub(r"\s+", " ", match.group("section_sign")).strip()
        else:
            kind = match.group("kind").title()
            label = match.group("label")
            if kind in {"Section", "Article"} and not label:
                continue
            canonical = " ".join(part for part in (kind, label) if part)
            if kind == "Article" and label:
                canonical = f"Article {label.upper()}"
        key = canonical.casefold()
        if key not in seen:
            references.append(
                {
                    "canonical_label": canonical,
                    "semantic_unit_id": targets.get(key),
                }
            )
            seen.add(key)
    return references


def _build_document_records(
    page_chunks: list[dict[str, Any]],
    source_file: str,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
) -> list[dict[str, Any]]:
    """Build indexable leaf records and link them to unsaved parent sections."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")

    base = _path_metadata(source_file)
    records: list[dict[str, Any]] = []
    stem = Path(source_file).stem
    sections = _build_primary_sections(page_chunks)
    semantic_ids = [f"{stem}|unit|{index}" for index in range(1, len(sections) + 1)]
    reference_targets = _reference_targets(sections, semantic_ids)
    for unit_index, section in enumerate(sections, start=1):
        # semantic_unit_id represents the complete primary section. Its body is
        # not duplicated in the index; only the leaf records below are persisted.
        semantic_unit_id = f"{stem}|unit|{unit_index}"
        leaves = _leaf_chunks(section, max_tokens, overlap_tokens)
        child_ids = [f"{semantic_unit_id}|chunk|{index}" for index in range(1, len(leaves) + 1)]
        semantic_text = _render(section.path, _body_text(section.body))

        for child_index, leaf in enumerate(leaves):
            chunk_id = child_ids[child_index]
            pages = list(leaf.pages)
            metadata = {
                **base,
                "title": base["document_title"],
                "chunk_id": chunk_id,
                "doc_id": chunk_id,
                "semantic_unit_id": semantic_unit_id,
                "source": source_file,
                "section_type": "appendix" if section.kind == "attachment" else (
                    "front_matter" if section.kind == "preamble" else "body"
                ),
                "section_path": list(leaf.section_path),
                "section_depth": len(leaf.section_path),
                "leaf_label": leaf.leaf_label,
                "split_strategy": leaf.split_strategy,
                "list_context": leaf.list_context,
                "cross_references": _cross_references(leaf.text, reference_targets),
                "continues_from_previous": bool(
                    child_index and leaves[child_index - 1].section_path == leaf.section_path
                ),
                "continues_in_next": bool(
                    child_index + 1 < len(leaves)
                    and leaves[child_index + 1].section_path == leaf.section_path
                ),
                "page_numbers": pages,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "semantic_unit_token_count": count_tokens(semantic_text),
                "token_count": count_tokens(leaf.text),
                "child_chunk_index": child_index + 1,
                "child_chunk_count": len(leaves),
                "prev_chunk_id": child_ids[child_index - 1] if child_index else None,
                "next_chunk_id": child_ids[child_index + 1] if child_index + 1 < len(child_ids) else None,
            }
            records.append({"id": chunk_id, "page_content": leaf.text, "metadata": metadata})
    return records


def _print_debug_documents(
    documents: list["Document"],
    excluded_artifacts: list[tuple[int, str]] | None = None,
) -> None:
    """Print leaf chunks grouped under their complete primary section."""
    for page, text in excluded_artifacts or ():
        print(f"Excluded interface artifact on page {page}: {text}")
    groups: dict[str, list["Document"]] = {}
    for document in documents:
        groups.setdefault(document.metadata["semantic_unit_id"], []).append(document)

    for section_index, (semantic_unit_id, children) in enumerate(groups.items(), start=1):
        metadata = children[0].metadata
        pages = sorted({page for child in children for page in child.metadata["page_numbers"]})
        print("=" * 80)
        print(f"SECTION {section_index}/{len(groups)}: {metadata['section_path'][0]}")
        print(f"Semantic unit: {semantic_unit_id}")
        print(f"Pages: {pages or 'unknown'}")
        print(f"Section tokens: {metadata['semantic_unit_token_count']}")
        print(f"Leaf chunks: {len(children)}")
        for child_index, document in enumerate(children, start=1):
            child_metadata = document.metadata
            print("-" * 80)
            print(f"CHUNK {child_index}/{len(children)}: {child_metadata['chunk_id']}")
            print(f"Path: {' > '.join(child_metadata['section_path'])}")
            print(f"Strategy: {child_metadata['split_strategy']}")
            print(f"Pages: {child_metadata['page_numbers'] or 'unknown'}")
            print(f"Tokens: {child_metadata['token_count']}")
            print(document.page_content)


def _add_pdf_layout_lines(pdf_path: Path, page_chunks: list[dict[str, Any]]) -> None:
    """Attach visual lines needed to recover flattened appendix table rows."""
    import pymupdf

    with pymupdf.open(pdf_path) as pdf:
        for index, chunk in enumerate(page_chunks):
            page_number = int((chunk.get("metadata") or {}).get("page_number") or index + 1)
            page = pdf[page_number - 1]
            visual_lines = []
            for block in page.get_text("dict").get("blocks", ()):
                for line in block.get("lines", ()):
                    text = "".join(span.get("text", "") for span in line.get("spans", ())).strip()
                    bbox = tuple(line.get("bbox", ()))
                    # CUAD source footers are outside the contract body and recur
                    # on every page; excluding the bottom margin prevents rows
                    # from being mislabeled as appendix entries.
                    if text and len(bbox) == 4 and bbox[1] < page.rect.height * 0.9:
                        visual_lines.append(
                            {
                                "text": text,
                                "bbox": bbox,
                                "page_width": page.rect.width,
                            }
                        )
            chunk["layout_lines"] = visual_lines


def load_pdf(
    relative_path: str,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
    debug: bool = False,
) -> list["Document"]:
    """Extract one project-relative PDF into hierarchy-aware leaf Documents."""
    root_path = Path(__file__).resolve().parents[3]
    pdf_path = root_path / relative_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import pymupdf4llm
    from langchain_core.documents import Document

    page_chunks = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        page_chunks=True,
        header=False,
        footer=False,
    )
    if not isinstance(page_chunks, list):
        raise TypeError("PyMuPDF4LLM did not return page chunks; version 1.28.0 is required")

    _add_pdf_layout_lines(pdf_path, page_chunks)
    artifacts = _interface_artifacts(page_chunks)
    records = _build_document_records(page_chunks, relative_path, max_tokens, overlap_tokens)
    documents = [Document(**record) for record in records]
    if debug:
        print(f"Built {len({doc.metadata['semantic_unit_id'] for doc in documents})} sections "
              f"and {len(documents)} chunks.")
        _print_debug_documents(documents, artifacts)
    return documents


def iter_corpus_pdf_paths(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> list[str]:
    """Return sorted project-relative paths for every PDF in a corpus directory."""
    root_path = Path(__file__).resolve().parents[3]
    corpus_path = root_path / corpus_dir
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")
    return sorted(
        path.relative_to(root_path).as_posix()
        for path in corpus_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_corpus(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
    max_documents: Optional[int] = None,
    debug: bool = False,
) -> list["Document"]:
    """Load all (or the first ``max_documents``) PDFs from a corpus."""
    pdf_paths = iter_corpus_pdf_paths(corpus_dir)
    if max_documents is not None:
        pdf_paths = pdf_paths[:max_documents]
    documents = []
    for relative_path in pdf_paths:
        documents.extend(load_pdf(relative_path, max_tokens, overlap_tokens, debug))
    return documents


if __name__ == "__main__":
    relative_path = "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
    load_pdf(relative_path, debug=True)
