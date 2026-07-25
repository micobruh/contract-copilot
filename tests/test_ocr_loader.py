import importlib.util
import io
import json
import re
import sys
import unittest
import gc
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/contract_copilot/indexer/ocr_loader.py"
SPEC = importlib.util.spec_from_file_location("contract_copilot_ocr_loader_test", MODULE_PATH)
ocr_loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ocr_loader
SPEC.loader.exec_module(ocr_loader)


class FakeTokenizer:
    def __init__(self):
        self.vocabulary = {}
        self.reverse = {}

    def _tokens(self, text):
        return list(re.finditer(r"\S+", text))

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.reverse[token_id] for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = []
        offsets = []
        for match in self._tokens(text):
            token = match.group(0)
            token_id = self.vocabulary.setdefault(token, len(self.vocabulary) + 1)
            self.reverse[token_id] = token
            ids.append(token_id)
            offsets.append(match.span())
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def page(text, number=1, classified_lines=(), layout_lines=()):
    boxes = []
    cursor = 0
    classified_lines = set(classified_lines)
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        if line.strip() in classified_lines:
            boxes.append({"class": "section-header", "pos": (cursor, end)})
        cursor = end
    return {
        "metadata": {"page_number": number},
        "text": text,
        "page_boxes": boxes,
        "layout_lines": list(layout_lines),
    }


class HierarchicalChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.tokenizer_patch = patch.object(ocr_loader, "_get_tokenizer", return_value=self.tokenizer)
        self.tokenizer_patch.start()

    def tearDown(self):
        self.tokenizer_patch.stop()

    def test_articles_sections_and_toc_rows(self):
        pages = [
            page("TABLE OF CONTENTS\nARTICLE 1 Definitions 1\nARTICLE 2 Services 4\n"),
            page(
                "ARTICLE I\nDEFINITIONS\nOpening text.\nSection 1.1 Scope. Work applies.\n",
                2,
                {"ARTICLE I", "DEFINITIONS", "Section 1.1 Scope. Work applies."},
            ),
            page(
                "ARTICLE II\nSERVICES\nSection 2.1 Hosting. Provider hosts.\n",
                3,
                {"ARTICLE II", "SERVICES", "Section 2.1 Hosting. Provider hosts."},
            ),
        ]

        sections = ocr_loader._build_primary_sections(pages)

        self.assertEqual([section.path[0] for section in sections], [
            "ARTICLE I Definitions",
            "ARTICLE II Services",
        ])
        self.assertFalse(any("TABLE OF CONTENTS" in " ".join(section.path) for section in sections))

    def test_recursive_subsections_preserve_parenthetical_prose(self):
        words = "one two three four five six seven eight"
        pages = [page(
            "1. Services.\n"
            f"Intro {words}.\n"
            f"1.1 Support. {words}.\n"
            "1.2 Hosting.\n"
            f"(a) Availability. {words}.\n"
            f"(b) Security. {words}.\n",
            classified_lines={"1. Services."},
        )]

        records = ocr_loader._build_document_records(
            pages,
            "data/raw/CUAD_v1/full_contract_pdf/Part_I/Test.pdf",
            max_tokens=22,
            overlap_tokens=4,
        )

        strategies = {record["metadata"]["split_strategy"] for record in records}
        paths = [record["metadata"]["section_path"] for record in records]
        self.assertIn("subsection", strategies)
        self.assertNotIn("nested_clause", strategies)
        self.assertTrue(any(path[-1].startswith("1.1") for path in paths))
        rendered = "\n".join(record["page_content"] for record in records)
        self.assertIn("(a) Availability.", rendered)
        self.assertIn("(b) Security.", rendered)
        self.assertTrue(all(record["metadata"]["token_count"] <= 22 for record in records))

    def test_token_fallback_overlaps_complete_sentences_and_counts_prefix(self):
        body = (
            "Alpha beta gamma ends. "
            "Delta epsilon zeta ends. "
            "Eta theta iota ends. "
            "Kappa lambda mu ends."
        )
        pages = [page(f"1. Long Section.\n{body}\n", classified_lines={"1. Long Section."})]

        records = ocr_loader._build_document_records(
            pages,
            "data/raw/CUAD_v1/full_contract_pdf/Part_I/Long.pdf",
            max_tokens=12,
            overlap_tokens=4,
        )

        self.assertGreaterEqual(len(records), 3)
        self.assertTrue(all(
            record["metadata"]["split_strategy"] == "sentence_window"
            for record in records
        ))
        self.assertTrue(all(record["metadata"]["token_count"] <= 12 for record in records))
        first_body = records[0]["page_content"].split("\n\n", 1)[1]
        second_body = records[1]["page_content"].split("\n\n", 1)[1]
        self.assertTrue(first_body.endswith("Delta epsilon zeta ends."))
        self.assertTrue(second_body.startswith("Delta epsilon zeta ends."))

    def test_oversized_sentence_uses_hard_token_windows(self):
        body = " ".join(f"word{index}" for index in range(50)) + "."
        pages = [page(f"1. Long Section.\n{body}\n", classified_lines={"1. Long Section."})]

        records = ocr_loader._build_document_records(
            pages,
            "data/raw/CUAD_v1/full_contract_pdf/Part_I/LongSentence.pdf",
            max_tokens=20,
            overlap_tokens=4,
        )

        first_body = records[0]["page_content"].split("\n\n", 1)[1].split()
        second_body = records[1]["page_content"].split("\n\n", 1)[1].split()
        self.assertEqual(first_body[-4:], second_body[:4])
        self.assertTrue(all(
            record["metadata"]["split_strategy"] == "hard_token_window"
            for record in records
        ))
        self.assertTrue(all(record["metadata"]["token_count"] <= 20 for record in records))

    def test_hard_windows_preserve_source_case_and_punctuation(self):
        body = (
            "Country-by-Country review preserves 2001/83/EC and DefinedTerm while "
            "additional words make this one legal sentence exceed the limit safely."
        )
        pages = [page(
            f"1. Fidelity.\n{body}\n",
            classified_lines={"1. Fidelity."},
        )]

        records = ocr_loader._build_document_records(
            pages,
            "contracts/Fidelity.pdf",
            max_tokens=12,
            overlap_tokens=3,
        )
        rendered = "\n".join(record["page_content"] for record in records)

        self.assertIn("Country-by-Country", rendered)
        self.assertIn("2001/83/EC", rendered)
        self.assertIn("DefinedTerm", rendered)
        self.assertTrue(all(
            record["metadata"]["split_strategy"] == "hard_token_window"
            for record in records
        ))

    def test_hard_windows_require_source_offsets(self):
        class NoOffsetTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                result = super().__call__(
                    text,
                    add_special_tokens=add_special_tokens,
                    return_offsets_mapping=return_offsets_mapping,
                )
                result.pop("offset_mapping", None)
                return result

        pages = [page(
            "1. Fidelity.\nOne two three four five six seven eight nine ten.\n",
            classified_lines={"1. Fidelity."},
        )]

        with self.assertRaisesRegex(ValueError, "offset mappings"):
            ocr_loader._build_document_records(
                pages,
                "contracts/NoOffsets.pdf",
                max_tokens=7,
                overlap_tokens=2,
                tokenizer=NoOffsetTokenizer(),
            )

    def test_explicit_tokenizer_is_used_for_every_chunk_limit(self):
        pages = [page(
            "1. Scope.\nAlpha beta gamma. Delta epsilon zeta.\n",
            classified_lines={"1. Scope."},
        )]

        with patch.object(
            ocr_loader,
            "_get_tokenizer",
            side_effect=AssertionError("default tokenizer should not be loaded"),
        ):
            records = ocr_loader._build_document_records(
                pages,
                "contracts/ExplicitTokenizer.pdf",
                max_tokens=8,
                overlap_tokens=2,
                tokenizer=FakeTokenizer(),
            )

        self.assertTrue(records)

    def test_recovers_glued_number_bare_appendix_and_bold_subsections(self):
        pages = [page(
            "# 1. Definitions\n"
            "Introductory language that makes the parent section large.\n"
            "**1.1 “Alpha”** means the first defined term and its complete body.\n"
            "**1.2 “Beta”** means the second defined term and its complete body.\n"
            "# 5.Term of this Agreement\n"
            "Term body.\n"
            "# Appendix\n"
            "Appendix body.\n"
        )]

        sections = ocr_loader._build_primary_sections(pages)
        self.assertEqual(
            [section.path[0] for section in sections],
            ["1. Definitions", "5. Term of this Agreement", "Appendix"],
        )
        children = ocr_loader._split_block_structurally(sections[0])
        self.assertEqual([child.path[-1] for child in children[1:]], [
            "1.1 “Alpha”",
            "1.2 “Beta”",
        ])
        self.assertIn("means the first defined term", ocr_loader._body_text(children[1].body))

    def test_parenthetical_clauses_stay_flat(self):
        clause = " ".join(f"word{index}" for index in range(25)) + "."
        pages = [page(
            f"1. Obligations.\n(a) {clause}\n(b) {clause}\n",
            classified_lines={"1. Obligations."},
        )]

        records = ocr_loader._build_document_records(
            pages, "contracts/Parentheticals.pdf", max_tokens=20, overlap_tokens=4
        )
        rendered = "\n".join(record["page_content"] for record in records)

        self.assertFalse(any(
            record["metadata"]["split_strategy"] == "nested_clause"
            for record in records
        ))
        self.assertIn("(a)", rendered)
        self.assertIn("(b)", rendered)

    def test_debug_output_groups_chunks_under_sections(self):
        metadata = {
            "semantic_unit_id": "Sample|unit|1",
            "section_path": ["1. Services"],
            "page_numbers": [1],
            "semantic_unit_token_count": 20,
            "chunk_id": "Sample|unit|1|chunk|1",
            "split_strategy": "section",
            "token_count": 8,
        }
        document = SimpleNamespace(metadata=metadata, page_content="## 1. Services\n\nBody.")
        output = io.StringIO()

        with redirect_stdout(output):
            ocr_loader._print_debug_documents([document], [(8, "Accept.")])

        rendered = output.getvalue()
        self.assertIn("Excluded interface artifact on page 8: Accept.", rendered)
        self.assertIn("SECTION 1/1: 1. Services", rendered)
        self.assertIn("CHUNK 1/1: Sample|unit|1|chunk|1", rendered)
        self.assertIn("Strategy: section", rendered)

    def test_rejects_caps_and_citations_without_duplicating_pages(self):
        pages = [
            page(
                "WARRANTIES ARE DISCLAIMED IN FULL\n"
                "C. Application Provider owns content.\n"
                "D. Provider distributes content.\n"
                "Section 7(d) applies to every party.\n"
                "Schedule A to this Agreement.\n"
                "First page text.\n"
            ),
            page("Second page text.\n", 2),
        ]

        records = ocr_loader._build_document_records(
            pages,
            "data/raw/CUAD_v1/full_contract_pdf/Part_I/Plain.pdf",
            max_tokens=256,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["metadata"]["section_path"], ["Document"])
        self.assertEqual(records[0]["metadata"]["page_numbers"], [1, 2])
        self.assertEqual(records[0]["page_content"].count("First page text."), 1)
        self.assertEqual(records[0]["page_content"].count("Second page text."), 1)

    def test_stable_ids_and_neighbor_metadata(self):
        pages = [page(
            "1. First.\nBody.\n2. Second.\nBody.\n",
            classified_lines={"1. First.", "2. Second."},
        )]
        records = ocr_loader._build_document_records(pages, "contracts/Sample.pdf")

        self.assertEqual(records[0]["id"], "Sample|unit|1|chunk|1")
        self.assertEqual(records[1]["id"], "Sample|unit|2|chunk|1")
        self.assertIsNone(records[0]["metadata"]["prev_chunk_id"])
        self.assertIsNone(records[0]["metadata"]["next_chunk_id"])

    def test_recovers_inline_and_table_wrapped_article_headers(self):
        pages = [page(
            "Body sentence. **ARTICLE II EQUIPMENT**\nEquipment text.\n"
            "|||**ARTICLE 3**||\nServices text.\n",
            classified_lines={"Body sentence. **ARTICLE II EQUIPMENT**", "|||**ARTICLE 3**||"},
        )]

        sections = ocr_loader._build_primary_sections(pages)

        self.assertEqual(
            [section.path[0] for section in sections],
            ["Preamble", "ARTICLE II EQUIPMENT", "ARTICLE 3"],
        )

    def test_list_children_inherit_exact_lead_ins_and_nested_context(self):
        pages = [page(
            "1. Eligibility.\n"
            "Applicants are rejected for the following reasons:\n"
            "- unlawful content\n"
            "- offensive content\n"
            "- Uses protected names in a domain, including:\n"
            "- CHASE\n"
            "- AARP\n",
            classified_lines={"1. Eligibility."},
        )]

        records = ocr_loader._build_document_records(
            pages, "contracts/Lists.pdf", max_tokens=18, overlap_tokens=4
        )
        contexts = {record["metadata"]["list_context"] for record in records}

        self.assertIn("Applicants are rejected for the following reasons:", contexts)
        self.assertIn("Uses protected names in a domain, including:", contexts)
        self.assertTrue(any(record["metadata"]["split_strategy"] == "list" for record in records))
        for record in records:
            context = record["metadata"]["list_context"]
            if context:
                self.assertIn(context, record["page_content"])

    def test_sentence_obligation_bullets_remain_separate(self):
        pages = [page(
            "2. Affiliate Responsibilities:\n"
            "- Affiliate must provide a current partner list.\n"
            "- Affiliate may not install spyware on any computer.\n"
            "- Affiliate may not overpay a sub-affiliate partner.\n",
            classified_lines={"2. Affiliate Responsibilities:"},
        )]

        records = ocr_loader._build_document_records(
            pages, "contracts/Duties.pdf", max_tokens=14, overlap_tokens=4
        )
        bodies = [record["page_content"] for record in records]

        self.assertTrue(any("install spyware" in body for body in bodies))
        self.assertTrue(any("overpay a sub-affiliate" in body for body in bodies))
        self.assertFalse(any("install spyware" in body and "overpay" in body for body in bodies))

    def test_exact_interface_controls_are_removed_but_prose_is_preserved(self):
        pages = [page("Accept.\nThe parties Accept. Terms stated here.\nAgree.\n")]

        records = ocr_loader._build_document_records(pages, "contracts/Controls.pdf")

        self.assertNotIn("\nAccept.\n", f"\n{records[0]['page_content']}\n")
        self.assertIn("The parties Accept. Terms stated here.", records[0]["page_content"])
        self.assertEqual(ocr_loader._interface_artifacts(pages), [(1, "Accept."), (1, "Agree.")])

    def test_attachment_layout_rows_use_confident_and_fallback_labels(self):
        layout = [
            {"text": "Appendix", "bbox": (50, 50, 90, 60), "page_width": 600},
            {"text": "Partner Restricted Trademark Terms", "bbox": (50, 70, 250, 80), "page_width": 600},
            {"text": "Sony Sony, ImageStation,", "bbox": (50, 90, 480, 100), "page_width": 600},
            {"text": "My Sony and Vaio", "bbox": (50, 100, 180, 110), "page_width": 600},
            {"text": "Unlabeled terms", "bbox": (50, 120, 180, 130), "page_width": 600},
            {"text": "Disney Restricted Key Words", "bbox": (50, 140, 240, 150), "page_width": 600},
            {"text": "cheap disney vacation", "bbox": (50, 150, 180, 160), "page_width": 600},
        ]
        pages = [page(
            "# Appendix\n" + "flattened attachment text " * 20,
            layout_lines=layout,
        )]

        records = ocr_loader._build_document_records(
            pages, "contracts/Appendix.pdf", max_tokens=20, overlap_tokens=4
        )
        paths = [record["metadata"]["section_path"] for record in records]

        self.assertTrue(any(path[-1] == "Sony" for path in paths))
        self.assertTrue(any(path[-1].startswith("Appendix entry") for path in paths))
        self.assertTrue(any(path[-1] == "Disney Restricted Key Words" for path in paths))
        sony = next(record for record in records if record["metadata"]["section_path"][-1] == "Sony")
        self.assertIn("My Sony and Vaio", sony["page_content"])

    def test_ordinary_exhibit_does_not_use_attachment_row_parser(self):
        layout = [
            {"text": "Exhibit 10.4", "bbox": (50, 50, 130, 60), "page_width": 600},
            {"text": "Agreement preamble.", "bbox": (50, 70, 200, 80), "page_width": 600},
            {"text": "1. Definitions", "bbox": (50, 90, 150, 100), "page_width": 600},
            {"text": "Definition body.", "bbox": (50, 110, 170, 120), "page_width": 600},
        ]
        pages = [page(
            "# Exhibit 10.4\nAgreement preamble.\n# 1. Definitions\nDefinition body.\n",
            layout_lines=layout,
        )]

        records = ocr_loader._build_document_records(pages, "contracts/Exhibit.pdf")
        rendered = "\n".join(record["page_content"] for record in records)

        self.assertFalse(any(
            record["metadata"]["split_strategy"] == "attachment_entry"
            for record in records
        ))
        self.assertEqual(rendered.count("Agreement preamble."), 1)
        self.assertEqual(rendered.count("Definition body."), 1)

    def test_bulleted_bold_legal_headings_are_primary_sections(self):
        pages = [page(
            "# 1. DEFINITIONS\nDefinition body.\n"
            "- **2. ASSIGNMENT** Assignment body.\n"
            "- **3. LICENSE** License body.\n"
            "- **4. PAYMENT** Payment body.\n"
            "- **5. TERM** Term body.\n"
            "- **6. GRANT** Grant body.\n"
        )]

        sections = ocr_loader._build_primary_sections(pages)

        self.assertEqual(
            [section.labels[0] for section in sections],
            ["1.", "2.", "3.", "4.", "5.", "6."],
        )
        self.assertNotIn("list", {section.kind for section in sections})

    def test_inline_bold_cross_reference_is_not_a_heading(self):
        pages = [page(
            "# 1. Definitions\n"
            "1.33 VerticalNet Content means the content described in\n"
            "**Section 3.1** **_[VERTICALNET CONTENT]_** .\n"
            "# 2. Services\nServices body.\n"
        )]

        records = ocr_loader._build_document_records(pages, "contracts/Citations.pdf")

        self.assertFalse(any(
            path[-1].startswith("Section 3.1")
            for record in records
            for path in [record["metadata"]["section_path"]]
        ))
        definition = next(
            record for record in records
            if record["metadata"]["section_path"][0].startswith("1.")
        )
        self.assertIn("Section 3.1", definition["page_content"])

    def test_line_cleaning_removes_presentation_markup_and_boundary_page_numbers(self):
        pages = [page(
            "12\n# <u>Terms</u><br><mark>Here</mark>\nBody\n7\nMore body\n34\n"
        )]

        lines = ocr_loader._page_lines(pages)
        texts = [line.text for line in lines if line.text]

        self.assertEqual(texts, ["# Terms Here", "Body", "7", "More body"])

    def test_bold_heading_keeps_abbreviated_title(self):
        pages = [page(
            "# ARTICLE 1 DEFINITIONS\n"
            "**1.28 “E.U. Major Countries”** means France, Germany, Italy, Spain, "
            "and the United Kingdom.\n"
            "**1.29 “FDA”** means the United States Food and Drug Administration "
            "and any successor authority with the same function.\n"
        )]

        records = ocr_loader._build_document_records(
            pages,
            "contracts/Abbreviations.pdf",
            max_tokens=24,
            overlap_tokens=4,
        )
        countries = next(
            record
            for record in records
            if record["metadata"]["section_path"][-1].startswith("1.28")
        )

        self.assertEqual(
            countries["metadata"]["section_path"][-1],
            "1.28 “E.U. Major Countries”",
        )
        self.assertIn("means France", countries["page_content"])

    def test_markdown_separators_do_not_become_chunks(self):
        pages = [page(
            "# 1. Restricted Names\n"
            "- BRITISH AIRWAYS\n"
            "|---|\n"
            "|•CASH PLUS|\n"
            "|---|\n"
            "- CHASE FREEDOM\n"
        )]

        records = ocr_loader._build_document_records(
            pages,
            "contracts/Separators.pdf",
        )
        rendered = "\n".join(record["page_content"] for record in records)

        self.assertNotIn("\n---", rendered)
        self.assertIn("BRITISH AIRWAYS", rendered)
        self.assertIn("CASH PLUS", rendered)
        self.assertIn("CHASE FREEDOM", rendered)

    def test_markdown_table_rows_are_atomic_sentence_units(self):
        pages = [page(
            "# 1. Pricing Notes\n"
            "|Note 1:|Purchase levels receive annual review.|\n"
            "|---|---|\n"
            "|Note 2:|Support staff remain certified.|\n"
        )]

        records = ocr_loader._build_document_records(
            pages,
            "contracts/TableNotes.pdf",
            max_tokens=11,
            overlap_tokens=2,
        )
        bodies = [record["page_content"] for record in records]

        self.assertTrue(any(
            "|Note 1:|Purchase levels receive annual review.|" in body
            for body in bodies
        ))
        self.assertTrue(any(
            "|Note 2:|Support staff remain certified.|" in body
            for body in bodies
        ))
        self.assertFalse(any(
            record["metadata"]["split_strategy"] == "hard_token_window"
            for record in records
        ))

    def test_cross_references_resolve_and_continuations_share_paths(self):
        pages = [page(
            "# 1. Definitions\nDefined terms.\n"
            "# 2. Duties\nSee Section 1 and Appendix. "
            "Alpha beta gamma ends. Delta epsilon zeta ends. Eta theta iota ends.\n"
            "# Appendix\nAttachment terms.\n"
        )]

        records = ocr_loader._build_document_records(
            pages, "contracts/References.pdf", max_tokens=12, overlap_tokens=4
        )
        duty_records = [
            record for record in records
            if record["metadata"]["section_path"][0] == "2. Duties"
        ]
        references = {
            ref["canonical_label"]: ref["semantic_unit_id"]
            for record in duty_records
            for ref in record["metadata"]["cross_references"]
        }

        self.assertEqual(references["Section 1"], "References|unit|1")
        self.assertEqual(references["Appendix"], "References|unit|3")
        self.assertTrue(duty_records[0]["metadata"]["continues_in_next"])
        self.assertTrue(duty_records[1]["metadata"]["continues_from_previous"])

    def test_all_explicit_reference_forms_and_unresolved_targets_are_retained(self):
        targets = {
            "article ii": "unit-article",
            "§ 3.1": "unit-section",
            "exhibit a": "unit-exhibit",
        }

        references = ocr_loader._cross_references(
            "See Article II, § 3.1, Exhibit A, and Schedule 2.", targets
        )
        resolved = {
            reference["canonical_label"]: reference["semantic_unit_id"]
            for reference in references
        }

        self.assertEqual(resolved["Article II"], "unit-article")
        self.assertEqual(resolved["§ 3.1"], "unit-section")
        self.assertEqual(resolved["Exhibit A"], "unit-exhibit")
        self.assertIsNone(resolved["Schedule 2"])


HAS_PDF_STACK = all(
    importlib.util.find_spec(module) is not None
    for module in ("pymupdf4llm", "langchain_core", "transformers")
)


@unittest.skipUnless(HAS_PDF_STACK, "optional PDF dependencies are not installed")
class CuadSmokeTests(unittest.TestCase):
    FILES = [
        "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/"
        "DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf",
        "data/raw/CUAD_v1/full_contract_pdf/Part_I/Development/"
        "AimmuneTherapeuticsInc_20200205_8-K_EX-10.3_11967170_EX-10.3_Development Agreement.pdf",
        "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/"
        "CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf",
        "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/"
        "SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement.pdf",
        "data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/"
        "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",
    ]

    @staticmethod
    def _normalized(text):
        text = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(re.findall(r"\w+", text))

    def test_representative_contracts(self):
        dataset_path = ROOT / "data/raw/CUAD_v1/CUAD_v1.json"
        if not dataset_path.exists():
            self.skipTest("CUAD annotations are not available")
        with dataset_path.open(encoding="utf-8") as dataset_file:
            dataset = json.load(dataset_file)["data"]
        selected_titles = {Path(path).stem for path in self.FILES}
        annotations = {
            item["title"]: [
                answer["text"]
                for paragraph in item["paragraphs"]
                for question in paragraph["qas"]
                for answer in question["answers"]
                if answer["text"]
            ]
            for item in dataset
            if item["title"] in selected_titles
        }
        del dataset
        gc.collect()

        answer_count = 0
        contained_answer_count = 0
        for relative_path in self.FILES:
            if not (ROOT / relative_path).exists():
                self.skipTest("CUAD dataset is not available")
            with self.subTest(relative_path=relative_path):
                documents = ocr_loader.load_pdf(relative_path)
                self.assertTrue(documents)
                self.assertTrue(all(document.metadata["token_count"] <= 256 for document in documents))
                self.assertTrue(all(document.metadata["section_path"] for document in documents))
                self.assertTrue(all(document.metadata["chunk_id"] for document in documents))
                self.assertTrue(all(document.metadata["page_numbers"] for document in documents))
                self.assertFalse(any(
                    "TABLE OF CONTENTS" in document.page_content
                    for document in documents
                ))
                self.assertFalse(any(
                    document.page_content.rstrip().endswith("\n\n---")
                    for document in documents
                ))

                normalized_chunks = [
                    self._normalized(document.page_content)
                    for document in documents
                ]
                for answer in annotations[Path(relative_path).stem]:
                    normalized_answer = self._normalized(answer)
                    if normalized_answer:
                        answer_count += 1
                        contained_answer_count += int(any(
                            normalized_answer in chunk
                            for chunk in normalized_chunks
                        ))

                for index, document in enumerate(documents):
                    metadata = document.metadata
                    previous = documents[index - 1] if index else None
                    following = documents[index + 1] if index + 1 < len(documents) else None
                    expected_previous = bool(
                        previous
                        and previous.metadata["semantic_unit_id"] == metadata["semantic_unit_id"]
                        and previous.metadata["section_path"] == metadata["section_path"]
                    )
                    expected_next = bool(
                        following
                        and following.metadata["semantic_unit_id"] == metadata["semantic_unit_id"]
                        and following.metadata["section_path"] == metadata["section_path"]
                    )
                    self.assertEqual(metadata["continues_from_previous"], expected_previous)
                    self.assertEqual(metadata["continues_in_next"], expected_next)
                if "CreditcardscomInc" in relative_path:
                    primary_paths = {document.metadata["section_path"][0] for document in documents}
                    self.assertIn("5. Term of this Agreement", primary_paths)
                    self.assertIn("Appendix", primary_paths)
                    self.assertFalse(any("Accept." in document.page_content for document in documents))

                    section_one = [
                        document for document in documents
                        if document.metadata["section_path"][0].startswith("1. Enrollment")
                    ]
                    section_two = [
                        document for document in documents
                        if document.metadata["section_path"][0] == "2. Affiliate Responsibilities"
                    ]
                    self.assertTrue(any(document.metadata["list_context"] for document in section_one))
                    self.assertTrue(all(
                        not document.metadata["list_context"]
                        or document.metadata["list_context"] in document.page_content
                        for document in section_one
                    ))
                    self.assertTrue(section_two)
                    self.assertTrue(all(document.metadata["split_strategy"] == "list" for document in section_two))
                    self.assertFalse(any(
                        "spyware" in document.page_content.lower()
                        and "higher referral fees" in document.page_content.lower()
                        for document in section_two
                    ))

                    appendix_labels = {
                        document.metadata["section_path"][-1]
                        for document in documents
                        if document.metadata["section_path"][0] == "Appendix"
                    }
                    self.assertTrue({
                        "Chase", "AARP", "Amazon", "Borders", "Sony", "Disney",
                        "United", "Continental",
                    }.issubset(appendix_labels))

                    references = {
                        reference["canonical_label"]: reference["semantic_unit_id"]
                        for document in documents
                        for reference in document.metadata["cross_references"]
                    }
                    self.assertIsNotNone(references.get("Section 4"))
                    self.assertIsNotNone(references.get("Appendix"))
                if "AimmuneTherapeuticsInc" in relative_path:
                    rendered = "\n".join(
                        document.page_content for document in documents
                    )
                    self.assertIn("country-by-country", rendered)
                    self.assertIn("2001/83/EC", rendered)
                    self.assertTrue(any(
                        document.metadata["section_path"][-1]
                        == "1.28 “E.U. Major Countries”"
                        for document in documents
                    ))
                if "CybergyHoldingsInc" in relative_path:
                    self.assertTrue(any(
                        "Note 6" in document.page_content
                        and "minimums defined by each Purchase Level"
                        in document.page_content
                        for document in documents
                    ))
                # Avoid retaining one extracted contract while PyMuPDF builds
                # the next; several representative files are hundreds of pages.
                del documents
                gc.collect()

        # This catches extraction/chunking regressions without requiring an
        # embedding model or conflating retrieval quality with generation.
        self.assertEqual(answer_count, 175)
        self.assertGreaterEqual(contained_answer_count, 160)


if __name__ == "__main__":
    unittest.main()
