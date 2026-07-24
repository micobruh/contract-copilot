import importlib.util
import io
import re
import sys
import unittest
import gc
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
            "Preamble",
            "ARTICLE I Definitions",
            "ARTICLE II Services",
        ])
        self.assertNotIn("ARTICLE 1 Definitions 1", " ".join(sections[0].path))

    def test_recursive_subsections_and_nested_clauses(self):
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
        self.assertIn("nested_clause", strategies)
        self.assertTrue(any(path[-1].startswith("1.1") for path in paths))
        self.assertTrue(any(path[-1].startswith("(a)") for path in paths))
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
        self.assertTrue(all(record["metadata"]["split_strategy"] == "token_fallback" for record in records))
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
        self.assertTrue(all(record["metadata"]["token_count"] <= 20 for record in records))

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

    def test_long_parenthetical_clause_stays_in_body(self):
        clause = " ".join(f"word{index}" for index in range(25)) + "."
        pages = [page(
            f"1. Obligations.\n(a) {clause}\n(b) {clause}\n",
            classified_lines={"1. Obligations."},
        )]

        section = ocr_loader._build_primary_sections(pages)[0]
        children = ocr_loader._split_block_structurally(section)

        self.assertEqual([child.path[-1] for child in children], ["(a)", "(b)"])
        self.assertIn("word24.", ocr_loader._body_text(children[0].body))

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

    def test_representative_contracts(self):
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
                # Avoid retaining one extracted contract while PyMuPDF builds
                # the next; several representative files are hundreds of pages.
                del documents
                gc.collect()


if __name__ == "__main__":
    unittest.main()
