## CS 294-264 HW1 — ReAct SWE Agent (Submission README)

### Contents
- Core agent: `cs294-264-hw-FA25/agent.py` (message tree, backtracking, tools, run loop)
- Response parser with doctests: `cs294-264-hw-FA25/response_parser.py`
- LLM wrapper (OpenAI Responses API): `cs294-264-hw-FA25/llm.py`
- Env tools: `cs294-264-hw-FA25/envs.py`
- Runner/CLI: `cs294-264-hw-FA25/run_agent.py`
- Results: `results_baseline`, `results_final_iter_2`, `results_final_iter_3`, `results_iter_next`

### Prereqs (brief)
- Docker Desktop running (`docker --version`, `docker run --rm hello-world`)
- Python 3.11 (Conda recommended)
- OpenAI key available to the process (`OPENAI_API_KEY`)

### Quick setup
```bash
conda create -y -n cs294-hw1 python=3.11
conda activate cs294-hw1
python -m pip install -U uv || true
uv pip install -r cs294-264-hw-FA25/requirements.txt || \
  python -m pip install -r cs294-264-hw-FA25/requirements.txt
# export OPENAI_API_KEY=... (or mini-swe-agent .env)
```

### Baseline run (no backtracking/tools/custom instructor; step cap 100)
```bash
python cs294-264-hw-FA25/run_agent.py \
  --subset cs294 --split test \
  -o results_baseline \
  --model gpt-5-mini \
  --max-steps 100 \
  --no-backtrack \
  --no-optional-tools \
  --no-debug
```
Evaluate baseline predictions:
```bash
python -m swebench.harness.run_evaluation \
  --dataset_name lynnliu030/swebench-eval-subset \
  --predictions_path ./results_baseline/preds.json \
  --max_workers 8 \
  --run_id baseline
```
Outputs:
- Predictions: `results_baseline/preds.json`
- Report: `gpt-5-mini.baseline.json`

### Improved run (with backtracking, optional tools, optimized instructor)
```bash
python cs294-264-hw-FA25/run_agent.py \
  --subset cs294 --split test \
  -o results_final_iter_2 \
  --model gpt-5-mini \
  --max-steps 100 \
  --backtrack \
  --optional-tools \
  --guard-empty-diff \
  --debug \
  --use-default-instructor
```
Evaluate improved predictions (adjust path per run):
```bash
python -m swebench.harness.run_evaluation \
  --dataset_name lynnliu030/swebench-eval-subset \
  --predictions_path ./results_final_iter_2/preds.json \
  --max_workers 8 \
  --run_id final_iter_2
```
Outputs:
- Predictions: `<run_dir>/preds.json`
- Report: `gpt-5-mini.<run_id>.json`

### Notes on flags
- `--backtrack`: enables `add_instructions_and_backtrack`.
- `--optional-tools`: registers extra env tools (edit helpers, grep, tests, syntax).
- `--guard-empty-diff`: require a non-empty staged `git diff` before `finish`.
- `--use-default-instructor`: optimized instructor prompt from `run_agent.py`.
- `--max-steps 100`: assignment step cap.

### Repro summary
- Model: `gpt-5-mini` via Responses API
- Dataset: `lynnliu030/swebench-eval-subset`
- Agent loop + tools per files above
- Result dirs: `results_baseline`, `results_final_iter_2`, `results_final_iter_3`, `results_iter_next`

### Packaging
- Submit code, `preds.json` and trajectory folders, and reports `gpt-5-mini.*.json`.
- Include a short PDF write-up; zip per assignment instructions.

### Troubleshooting
- Empty/invalid patches: use `--guard-empty-diff` on improved runs.
- Syntax errors after edits: use `syntax_check` tool (with `--optional-tools`).
- Docker not found: install Docker Desktop and restart shell.
