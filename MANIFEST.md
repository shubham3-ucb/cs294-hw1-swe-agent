# Submission Manifest — CS 294-264 HW1 ReAct SWE Agent

## Environment
- Python: 3.11
- mini-swe-agent: 1.13.3
- Dataset: lynnliu030/swebench-eval-subset (test split)
- Model: gpt-5-mini (OpenAI Responses API, reasoning effort=medium)
- Max steps: 100

## Artifacts Included
- cs294-264-hw-FA25/ — source code (agent, env tools, LLM, parser, runner)
- SUBMISSION_README.md — run instructions and evaluation steps
- MANIFEST.md — this file (inventory + repro mapping)
- results_baseline/preds.json — baseline predictions
- results_final_improved/preds.json — improved predictions
- gpt-5-mini.baseline.json — baseline evaluation report
- gpt-5-mini.final_improved.json — improved evaluation report
- results_*/*/*.traj.json — per-instance trajectories (tool calls and outputs)

## Reproduction Commands
Baseline (pure):
```bash
python cs294-264-hw-FA25/run_agent.py \
  --subset cs294 --split test \
  -o results_baseline \
  --model gpt-5-mini \
  --max-steps 100 \
  --no-backtrack \
  --no-optional-tools \
  --no-debug

python -m swebench.harness.run_evaluation \
  --dataset_name lynnliu030/swebench-eval-subset \
  --predictions_path ./results_baseline/preds.json \
  --max_workers 8 \
  --run_id baseline
```

Improved (with tools/backtracking/default instructor/guard):
```bash
python cs294-264-hw-FA25/run_agent.py \
  --subset cs294 --split test \
  -o results_final_improved \
  --model gpt-5-mini \
  --max-steps 100 \
  --backtrack \
  --optional-tools \
  --guard-empty-diff \
  --debug \
  --use-default-instructor

python -m swebench.harness.run_evaluation \
  --dataset_name lynnliu030/swebench-eval-subset \
  --predictions_path ./results_final_improved/preds.json \
  --max_workers 8 \
  --run_id final_improved
```

## Mapping: Predictions → Reports
- results_baseline/preds.json → gpt-5-mini.baseline.json
- results_final_improved/preds.json → gpt-5-mini.final_improved.json

## Notes
- Baseline registers only `run_bash_cmd` (plus mandatory `finish`); no backtracking, no optional tools, no default instructor, no diff guard.
- Improved run registers optional tools in `cs294-264-hw-FA25/envs.py` and uses an optimized instructor prompt.
