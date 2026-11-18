# Bounded catalog prototype and Cursor test handoff

Use `build_retrieve/main.py retrieve --open_ended`. This feature does not use or
modify `answer_open.py`, SIB-GRPO, or L1/L2/L3 construction. It reuses built memory.
`--retrieval_mode hierarchical` remains the default and preserves the existing
open-ended L3 -> L2 -> L1 cascade. `--retrieval_mode catalog` opts into the new loop.

## What is implemented

- A question-independent chronological index over **all visible L1 and L2
  records**, including L1 records discarded by L2 grouping. L3 descriptions are
  attached to their linked L2 records. Source memory is not edited.
- The entire rendered root catalog (headings, time spans, IDs, summaries) fits
  `--catalog_token_budget`, default 2,000, measured with the model tokenizer.
- External pages have their own token budget and at most `catalog_fanout`
  children. Extra levels keep every record reachable as history grows. This
  bounds root length, **not retrieval depth or total storage**.
- Full records, pages, and routing summaries are stored together in a
  content-addressed JSON snapshot under `MEMORY_DIR/VIDEO_ID/catalogs/`.
  Pages/records are external to the prompt until read. One artifact makes atomic
  cache publication simple; it is not a new video-memory representation.
- Leaf descriptions are explicitly shortened previews of existing text.
  Parent descriptions default to model-generated summaries (`model`).
  `extractive` uses shortened concatenations instead, avoiding summarization
  calls; it may hide distinctive details and is a cheap diagnostic baseline.
  Full source text is retained and can be read with token-offset pagination.
- Budgets are enforced by shortening marked summaries while keeping pointers;
  impossible budgets raise an error rather than dropping catalog entries.
- The answer model receives the full root catalog at every turn and emits JSON
  `read`, `view`, or `answer` actions. `read` navigates pages or paginated records.
  `view` presents uniformly selected saved frames only after a record is read.
  There is no embedding prefilter, visual relevance sweep, or write operation.
- Invalid actions are reported to the model. Evidence IDs must reference records
  already read. This checks provenance, not whether an answer is semantically
  supported. The model can abstain. At most 8 action turns plus one final-answer
  turn are allowed by default; invalid responses consume turns too.
- Saved images are selected before loading, with per-request and overall unique
  frame budgets. No raw-video decode/recovery tool is added in this step.
- Each question starts fresh. Prior answers and gold annotations are not inputs.

## Temporal filtering and limitations

Use exactly the same cutoff for both retrieval modes. `--before_clip N` sets
`until_sec = (N + 1) * 30`, taking precedence over `--until_sec`.
L1/L2 records crossing the cutoff or missing an end time are excluded. L3 inherits
its linked L2 interval. **Filtering happens before catalog grouping or model
summarization**. Cache identity includes filtered record contents, cutoff, schema,
budgets, summary mode, model path, and tokenizer vocabulary/configuration.
If weights are replaced at the same path, use `--catalog_rebuild`.

This is retrospective filtering of existing records, not a true streaming builder.
Existing offline L2 segmentation may itself depend on later observations.
Untimed records are included only when no cutoff is specified. A cutoff snapshot
can be empty: the agent should abstain, not search future memory.

Chronological grouping and preview compression are hand-designed prototype
policies. No learned directory update policy is claimed. Catalog construction is
currently a separate compilation pass, not an online memory-manager operation.

The navigation history has a text/template token budget, separate from the root
budget. If it is exceeded, the loop returns an explicit context-budget abstention
without a generation call. Expanded visual patch tokens are **not** counted by
that text budget. Frames also remain in the subsequent context, so they may be
processed repeatedly. The existing MM-Mem backend handles image preprocessing;
test actual processor behavior and VRAM on the DGX. Use the normal model context
limit initially; do not assume a small `max_model_len` fits visual inputs simply
because the text fits.

## Local CPU checks

From the repository root (no model or dataset downloads):

```bash
python -m unittest discover -s tests -v
python -m compileall -q build_retrieve tests
python build_retrieve/main.py retrieve --help
python build_retrieve/main.py catalog --help
```

These tests use a character tokenizer and scripted model responses. They verify
contracts, not Qwen tokenizer/chat-template compatibility or task accuracy.

## DGX / Cursor instructions

1. Fetch `feat/bounded-memory-catalog`. Inspect `git status` and running GPU jobs
   before switching or starting inference. Preserve local edits and active runs;
   use a separate worktree if needed. Do not rebuild the expensive video memory.
2. Activate the existing working MM-Mem environment. Verify the paths below;
   they are inferred from the original fork, not inspected on the DGX.
3. Run the CPU checks above. Then compile one catalog and answer one question.
4. Start with `--catalog_summary_mode extractive` to check plumbing. Then use
   `model` summaries for the actual prototype. Report both modes separately.

```bash
cd /home/hzhao/long/MM-Mem

# Set MEMORY_ROOT to the directory directly containing study_17/ with its JSONLs.
# Some existing runs may use output/study_17/study_17/*.jsonl; in that case use
# MEMORY_ROOT=/home/hzhao/long/MM-Mem/output/study_17.
MEMORY_ROOT=/home/hzhao/long/MM-Mem/output
MODEL_PATH=/home/hzhao/long/weights/Qwen3-VL-8B-Instruct
VIDEO_PATH=/actual/path/to/study_17.mp4

python build_retrieve/main.py catalog \
  --memory_dir "$MEMORY_ROOT" --video_id study_17 \
  --base_model_path "$MODEL_PATH" \
  --catalog_summary_mode extractive --catalog_token_budget 2000

python build_retrieve/main.py retrieve \
  --video_path "$VIDEO_PATH" --memory_dir "$MEMORY_ROOT" --video_id study_17 \
  --base_model_path "$MODEL_PATH" --open_ended --retrieval_mode catalog \
  --question "Where is the alcohol placed?" \
  --catalog_summary_mode extractive --catalog_token_budget 2000 \
  --output_file /tmp/mm_mem_catalog_smoke.json
```

For a benchmark question, also pass its actual `--before_clip` value. The commands
above intentionally omit a made-up cutoff. Output files are created exclusively:
choose new filenames for later tests rather than overwriting previous results.

For a model-summary run, change `extractive` to `model`; the first question may
compile the catalog, subsequent questions at the same cutoff reuse it. The
standalone extractive compiler loads only the tokenizer; answering loads the VLM.
Model-summary compilation loads the VLM and makes text-only summary calls.

Compare with the same question/cutoff/model using:

```bash
python build_retrieve/main.py retrieve \
  --video_path "$VIDEO_PATH" --memory_dir "$MEMORY_ROOT" --video_id study_17 \
  --base_model_path "$MODEL_PATH" --open_ended --retrieval_mode hierarchical \
  --question "Where is the alcohol placed?" \
  --output_file /tmp/mm_mem_hierarchical_smoke.json
```

The historical hierarchical path needs its existing embedding/reranker weights;
pass `--embedding_model_name` and `--reranker_model_name` with their existing
local paths if Hugging Face downloads are unavailable. Catalog mode does not use
them. `--catalog_max_visual_frames 0` provides an explicit text-only ablation
through the same main.py entry point.

## What to report back

- Active commit, actual memory layout, L1/L2/L3 record counts, model/backend,
  package versions, real question IDs and cutoffs, and command lines.
- Catalog token count <= budget; number/depth of external pages; a few root
  entries; whether the medicine-box/alcohol evidence is discoverable.
- Cache miss then hit for the same cutoff; no future content in an earlier
  snapshot after running a later cutoff first.
- Complete result JSON and traceback for any failures; verify JSON action
  compliance with the actual Qwen processor and generation backend.
- One text-sufficient question and one question requiring visual detail. Check
  that the former makes no `view` calls and the latter can request useful frames.
- Accuracy is a separate check: correct schema/citations do not prove correctness.

Result metrics distinguish catalog-building calls from answering calls, tool
calls, unique frames from repeated frame presentations, cumulative text/template
input tokens, and encoded output-text tokens. These are **not** exact multimodal
token usage or GPU allocation measurements. `query_total_seconds` includes
catalog setup/model initialization; `answer_seconds` covers the answering loop.
Cold and warm runs must be reported separately. The original hierarchical mode
does not yet have matching detailed cost instrumentation; do not infer a fair
efficiency comparison from catalog metrics alone.

## Deferred

Training, changing ADD_NEW/MERGE/DISCARD, online catalog rewriting, semantic graph
edge changes, compressed-video storage, cross-question memory updates, and GUI
adapters are intentionally deferred to separate changes/experiments.
