"""CPU-only contract tests; run: python -m unittest discover -s tests -v."""
import copy
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import runpy
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build_retrieve"))
from catalog import (cached_catalog, compile_catalog, load_records, render_entries,
                     token_count, validate_catalog)
from catalog_retrieve import get_catalog, navigate_catalog, select_frame_paths
from config import get_args, build_config


class CharacterTokenizer:
    """Deliberately counts punctuation/whitespace too; no dependency or download."""
    name_or_path = "character-test"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def records(n=1):
    return {f"L2:{i}": {"id": f"L2:{i}", "layer": "L2", "index": i,
                         "start_sec": i * 10, "end_sec": (i + 1) * 10,
                         "text": f"Episode {i}: alcohol is in the medicine box. " * 10,
                         "l1_ids": [], "frame_paths": [], "frame_timestamps": []}
            for i in range(n)}


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()

    def test_bounded_tree_keeps_all_records(self):
        source = records(100)
        catalog = compile_catalog(source, self.tok, budget=400, page_budget=500,
                                  fanout=4, summary_tokens=70)
        self.assertLessEqual(token_count(self.tok, render_entries("Memory catalog", catalog["root"])), 400)
        self.assertEqual(source, catalog["records"])
        validate_catalog(catalog, self.tok)
        self.assertGreater(len(catalog["pages"]), 1)

    def test_invalid_pointer_and_cycle_rejected(self):
        catalog = compile_catalog(records(20), self.tok, fanout=4)
        bad = copy.deepcopy(catalog)
        bad["root"][0]["id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Dangling"):
            validate_catalog(bad, self.tok)
        bad = copy.deepcopy(catalog)
        pid = next(iter(bad["pages"]))
        bad["pages"][pid] = [{**bad["root"][0], "id": pid}]
        with self.assertRaises(ValueError):
            validate_catalog(bad, self.tok)

    def test_tiny_budget_fails_without_dropping_pointers(self):
        with self.assertRaises(ValueError):
            compile_catalog(records(8), self.tok, budget=64)

    def test_empty_catalog(self):
        catalog = compile_catalog({}, self.tok)
        self.assertEqual(catalog["root"], [])
        validate_catalog(catalog, self.tok)

    def _memory(self, folder):
        root = Path(folder) / "video"
        root.mkdir()
        for name, rows in {
            "l1_memory.jsonl": [
                {"l1_index": 0, "start_sec": 0, "end_sec": 10, "caption": "early clue"},
                {"l1_index": 1, "start_sec": 10, "end_sec": 40, "caption": "FUTURE"},
                {"l1_index": 2, "caption": "UNTIMED"}],
            "l2_memory.jsonl": [
                {"l2_index": 0, "start_sec": 0, "end_sec": 10, "caption": "early episode", "constituent_l1_indices": [0, 1]},
                {"l2_index": 1, "start_sec": 10, "end_sec": 40, "caption": "FUTURE"}],
            "l3_memory.jsonl": [
                {"l2_index": 0, "entities": [{"entity name": "early entity"}]},
                {"l2_index": 1, "entities": [{"entity name": "FUTURE"}]}],
        }.items():
            (root / name).write_text("\n".join(json.dumps(row) for row in rows))
        return root

    def test_cutoff_before_summary_and_l3_inheritance(self):
        with tempfile.TemporaryDirectory() as folder:
            self._memory(folder)
            source = load_records(folder, "video", until_sec=30)
            self.assertEqual(set(source), {"L1:0", "L2:0"})
            self.assertIn("early entity", source["L2:0"]["text"])
            self.assertEqual(source["L2:0"]["l1_ids"], ["L1:0"])
            summaries = []
            def summarize(text, limit):
                summaries.append(text)
                return "early summary"
            # More records force a parent summary.
            extended = {**source, **{f"L2:{i}": {**source["L2:0"], "id": f"L2:{i}"} for i in range(2, 5)}}
            compile_catalog(extended, self.tok, fanout=2, summarizer=summarize)
            self.assertTrue(summaries)
            self.assertNotIn("FUTURE", " ".join(summaries))
            self.assertNotIn("UNTIMED", json.dumps(source))

    def test_cache_separates_cutoffs_modes_budgets_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._memory(folder)
            _, path1, hit = cached_catalog(folder, "video", self.tok, until_sec=10)
            self.assertFalse(hit)
            _, path2, hit = cached_catalog(folder, "video", self.tok, until_sec=10)
            self.assertEqual(path1, path2)
            self.assertTrue(hit)
            late, path3, _ = cached_catalog(folder, "video", self.tok, until_sec=40)
            self.assertNotEqual(path1, path3)
            self.assertIn("FUTURE", json.dumps(late))
            _, path4, _ = cached_catalog(folder, "video", self.tok, until_sec=10, budget=1000)
            self.assertNotEqual(path1, path4)
            with (root / "l1_memory.jsonl").open("a") as handle:
                handle.write('\n' + json.dumps({"l1_index": 9, "start_sec": 0, "end_sec": 1, "caption": "added"}))
            _, path5, _ = cached_catalog(folder, "video", self.tok, until_sec=10)
            self.assertNotEqual(path1, path5)

    def _run(self, responses, catalog=None, **kwargs):
        sent = []
        responses = iter(responses)
        def ask(messages):
            sent.append(copy.deepcopy(messages))
            return next(responses)
        catalog = catalog or compile_catalog(records(), self.tok)
        result = navigate_catalog(catalog, self.tok, "Where is alcohol?", ask, **kwargs)
        return result, sent

    def test_text_answer_never_loads_images_and_always_has_full_catalog(self):
        def forbidden(*args):
            self.fail("Text-only answer loaded frames")
        result, sent = self._run([
            '{"action":"read","id":"L2:0"}',
            '{"action":"answer","answer":"In the medicine box.","evidence_ids":["L2:0"]}'
        ], frame_loader=forbidden)
        self.assertEqual(result["final_answer"], "In the medicine box.")
        self.assertEqual(result["metrics"]["frames_presented"], 0)
        self.assertEqual(result["metrics"]["answer_llm_calls"], 2)
        self.assertEqual(sent[0][0], sent[1][0])
        self.assertIn("Memory catalog", sent[1][0]["content"])

    def test_page_navigation_exposes_children(self):
        catalog = compile_catalog(records(3), self.tok, fanout=2)
        page = catalog["root"][0]["id"]
        rid = catalog["pages"][page][0]["id"]
        result, _ = self._run([
            json.dumps({"action": "read", "id": page}),
            json.dumps({"action": "read", "id": rid}),
            json.dumps({"action": "answer", "answer": "Found", "evidence_ids": [rid]}),
        ], catalog=catalog)
        self.assertEqual(result["evidence_ids"], [rid])

    def test_invalid_json_and_unknown_ids_are_recoverable(self):
        result, _ = self._run(['bad', '{"action":"read","id":"L2:999"}',
                               '{"action":"answer","answer":"Unknown.","abstain":true}'])
        self.assertTrue(result["abstained"])
        self.assertIn("error", result["retrieval_trace"][0])
        self.assertIn("error", result["retrieval_trace"][1])

    def test_answer_cannot_cite_unread_records(self):
        result, _ = self._run([
            '{"action":"answer","answer":"guess","evidence_ids":["L2:0"]}',
            '{"action":"answer","answer":"Unknown.","abstain":true}',
        ])
        self.assertIn("error", result["retrieval_trace"][0])
        self.assertTrue(result["abstained"])

    def test_visual_budget_and_repeat_presentation_accounting(self):
        result, sent = self._run([
            '{"action":"view","id":"L2:0"}',
            '{"action":"read","id":"L2:0"}',
            '{"action":"view","id":"L2:0"}',
            '{"action":"view","id":"L2:0"}',
            '{"action":"answer","answer":"box","evidence_ids":["L2:0"]}',
        ], frame_loader=lambda record, limit: ["frame.jpg"] * limit,
            max_frames=2, total_frames=1)
        self.assertIn("error", result["retrieval_trace"][0])
        self.assertIn("error", result["retrieval_trace"][3])
        self.assertEqual(result["metrics"]["unique_frames"], 1)
        self.assertEqual(result["metrics"]["frames_presented"], 2)

    def test_record_pagination(self):
        result, _ = self._run([
            '{"action":"read","id":"L2:0","offset":0}',
            '{"action":"read","id":"L2:0","offset":30}',
            '{"action":"answer","answer":"Unknown.","abstain":true}',
        ], read_tokens=30)
        first, second = result["retrieval_trace"][:2]
        self.assertEqual(first["result"]["next_offset"], 30)
        self.assertEqual(second["result"]["next_offset"], 60)
        self.assertNotEqual(first["result"]["text"], second["result"]["text"])

    def test_round_limit_forces_final_answer(self):
        result, _ = self._run(['bad', '{"action":"read","id":"L2:0"}'], max_rounds=1)
        self.assertEqual(result["stop_reason"], "invalid_final_response")
        self.assertEqual(result["metrics"]["answer_llm_calls"], 2)

    def test_context_limit_prevents_generation(self):
        result, sent = self._run([], context_tokens=10)
        self.assertEqual(result["stop_reason"], "context_budget")
        self.assertEqual(sent, [])

    def test_frame_paths_stay_in_memory_root_and_sample_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "video" / "L2_frames" / "l2_0000"
            base.mkdir(parents=True)
            names = [f"{i}.jpg" for i in range(10)]
            for name in names:
                (base / name).touch()
            record = records()["L2:0"]
            record["frame_paths"] = names
            record["frame_timestamps"] = list(range(10))
            selected = select_frame_paths(record, folder, "video", 3)
            self.assertEqual(len(selected), 3)
            self.assertTrue(selected[0].endswith("0.jpg"))
            self.assertTrue(selected[-1].endswith("9.jpg"))
            record["frame_paths"] = ["../../../../escape.jpg"]
            with self.assertRaises(ValueError):
                select_frame_paths(record, folder, "video", 3)

    def test_cli_defaults_and_catalog_options(self):
        common = ["main.py", "retrieve", "--video_path", "video.mp4", "--memory_dir", "memory",
                  "--video_id", "video", "--question", "Where?", "--open_ended"]
        with patch.object(sys, "argv", common):
            self.assertEqual(build_config(get_args()).retrieval_mode, "hierarchical")
        with patch.object(sys, "argv", common + ["--retrieval_mode", "catalog", "--catalog_token_budget", "500"]):
            cfg = build_config(get_args())
            self.assertEqual(cfg.retrieval_mode, "catalog")
            self.assertEqual(cfg.catalog_token_budget, 500)
        with patch.object(sys, "argv", ["main.py", "catalog", "--memory_dir", "memory", "--video_id", "video"]):
            self.assertEqual(get_args().command, "catalog")

    def test_main_retrieve_forwards_mode_cutoff_and_saves_result(self):
        calls = []
        model_module = types.ModuleType("models")
        model_module.ModelManager = lambda config: object()
        retrieve_module = types.ModuleType("memory_retrieve")
        def retrieve(**kwargs):
            calls.append(kwargs)
            return {"final_answer": "box", "stage": "catalog"}
        retrieve_module.hierarchical_retrieve_and_answer = retrieve
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "result.json")
            argv = ["main.py", "retrieve", "--video_path", "v.mp4", "--memory_dir", "m",
                    "--video_id", "v", "--question", "Where?", "--open_ended",
                    "--retrieval_mode", "catalog", "--before_clip", "2", "--until_sec", "999",
                    "--output_file", output]
            with patch.dict(sys.modules, {"models": model_module, "memory_retrieve": retrieve_module}), \
                    patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "build_retrieve/main.py"), run_name="__main__")
                with self.assertRaises(FileExistsError):
                    runpy.run_path(str(Path(__file__).resolve().parents[1] / "build_retrieve/main.py"), run_name="__main__")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["until_sec"], 90)
            self.assertEqual(calls[0]["config"].retrieval_mode, "catalog")
            self.assertTrue(calls[0]["open_ended"])
            self.assertEqual(json.loads(Path(output).read_text())["final_answer"], "box")

    def test_extractive_compilation_does_not_load_vlm(self):
        from config import MemoryConfig
        model_module = types.ModuleType("models")
        model_module.generate_text = lambda *args, **kwargs: self.fail("Unexpected model call")
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: self.tok)
        with tempfile.TemporaryDirectory() as folder:
            self._memory(folder)
            cfg = MemoryConfig(memory_dir=folder, catalog_summary_mode="extractive")
            manager = types.SimpleNamespace(get_base_model=lambda: self.fail("Loaded VLM"))
            with patch.dict(sys.modules, {"models": model_module, "transformers": transformers_module}):
                _, _, stats = get_catalog(cfg, manager, "video", until_sec=10)
                _, _, warm = get_catalog(cfg, manager, "video", until_sec=10)
            self.assertEqual(stats["catalog_build_llm_calls"], 0)
            self.assertTrue(warm["catalog_cache_hit"])


if __name__ == "__main__":
    unittest.main()
