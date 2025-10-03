#!/usr/bin/env python3
import concurrent.futures
import subprocess
from pathlib import Path

import typer
from datasets import load_dataset

from utils import save_traj, update_preds_file, remove_from_preds_file, get_sb_environment

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "cs294": "lynnliu030/swebench-eval-subset",
}

from agent import ReactAgent
from llm import OpenAIModel
from response_parser import ResponseParser
from envs import SWEEnvironment, DumbEnvironment

DEFAULT_INSTRUCTOR = (
    "Output EXACTLY ONE function call per step using the protocol shown below.\n\n"
    "Minimal, test-driven workflow:\n"
    "1) Run tests/logs:\n"
    "   - Prefer run_common_tests(). If project has custom entry, use run_bash_cmd to run it.\n"
    "   - Read ONLY the top traceback frame. Extract file and a tight line window (e.g., function).\n"
    "   - Confirm location via grep_repo(symbol) if needed.\n"
    "2) Inspect before editing:\n"
    "   - show_file_range(file_path, from_line, to_line) for the implicated block.\n"
    "3) Edit surgically:\n"
    "   - Use replace_in_file(file_path, from_line, to_line, content) to change JUST that block.\n"
    "   - Preserve EXACT leading indentation and surrounding context. Do NOT outdent a method or move it outside its class.\n"
    "   - Prefer editing inside a method body; do not change the def/class signature unless the error explicitly requires it.\n"
    "   - Do NOT edit tests unless required. Avoid cosmetic edits.\n"
    "4) Validate:\n"
    "   - First run syntax_check().\n"
    "   - Then re-run tests. If still failing, repeat from step 1 on the NEW top frame.\n"
    "5) Stage and verify diff:\n"
    "   - run_bash_cmd(\"git add -A\"); then call stage_and_diff(). Ensure diff is NON-EMPTY and includes the file from the top traceback.\n"
    "6) Finish:\n"
    "   - Only call finish after a non-empty staged diff and tests look correct. Summarize in 1–2 sentences.\n\n"
    "Protocol rules:\n"
    "- One function call only. If you accidentally emitted two, re-emit a single well-formed call next step.\n"
    "- After ----ARG---- put the arg name on its own line, then the value (can be multiline).\n"
    "- Never pass an empty command to run_bash_cmd. Keep changes minimal and relevant.\n"
    "- Do NOT create new files; only edit existing source files relevant to the failure."
)

def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str,
    max_steps: int,
    backtrack: bool,
    optional_tools: bool,
    instructor_text: str | None,
    debug: bool,
    guard_empty_diff: bool,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    
    # Avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    
    # Initialize the model and parser
    llm = OpenAIModel(ResponseParser.END_CALL, model_name)
    parser = ResponseParser()
    task = instance["problem_statement"]
    
    print(f"Processing instance {instance_id}")
    agent = None    
    result = ""
    
    try:
        # Initialize the environment
        env = SWEEnvironment(instance)
        # Initialize the agent
        agent = ReactAgent("swe-agent", parser, llm)
        
        # Debug: print basic repo context inside sandbox
        if debug:
            try:
                print("[DEBUG] PWD:")
                print(env.env.execute("pwd"))
                print("[DEBUG] Git toplevel:")
                print(env.env.execute("git rev-parse --show-toplevel"))
                print("[DEBUG] Untracked/modified files count:")
                print(env.env.execute("git status --porcelain | wc -l"))
            except Exception as e:
                print(f"[DEBUG] Failed to gather repo context: {e}")
        # Optionally set instructor text before registering tools
        if instructor_text:
            agent.set_instructions(instructor_text)

        # Register tools BEFORE running
        tools = [env.run_bash_cmd]
        if backtrack:
            tools.append(agent.add_instructions_and_backtrack)
        if optional_tools:
            if hasattr(env, "replace_in_file"):
                tools.append(env.replace_in_file)
            if hasattr(env, "show_file"):
                tools.append(env.show_file)
            if hasattr(env, "stage_and_diff"):
                tools.append(env.stage_and_diff)
            if hasattr(env, "show_file_range"):
                tools.append(env.show_file_range)
            if hasattr(env, "grep_repo"):
                tools.append(env.grep_repo)
            if hasattr(env, "run_common_tests"):
                tools.append(env.run_common_tests)
            if hasattr(env, "syntax_check"):
                tools.append(env.syntax_check)
        agent.add_functions(tools)
        # Run the agent
        output = agent.run(task, max_steps, guard_empty_diff=guard_empty_diff, debug=debug)
        
        # Generate patch for SWE-Bench
        result = env.generate_patch(output)
        if debug:
            print("[DEBUG] Generated patch preview (first 80 lines):")
            try:
                preview = env.env.execute("git diff --cached | sed -n '1,80p'")
                print(preview)
            except Exception as e:
                print(f"[DEBUG] Failed to preview patch: {e}")
        
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        
    finally:
        # Save the trajectory and update the predictions file
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            result=result,
            instance_id=instance_id,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        print(f"Completed instance {instance_id}, result: {result}")

@app.command(help="Run CS294 HW on subset of SWEBench instances.")
def main(
    subset: str = typer.Option("cs294", "--subset", help="SWEBench subset used or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output: str = typer.Option("outputs", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model_name: str = typer.Option("gpt-5-mini", "--model", help="Model used", rich_help_panel="Basic"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum number of steps", rich_help_panel="Basic"),
    limit: int = typer.Option(0, "--limit", help="Limit number of instances to process (0 = all)", rich_help_panel="Basic"),
    backtrack: bool = typer.Option(False, "--backtrack/--no-backtrack", help="Enable add_instructions_and_backtrack tool", rich_help_panel="Tools"),
    optional_tools: bool = typer.Option(False, "--optional-tools/--no-optional-tools", help="Enable optional tools like replace_in_file and show_file", rich_help_panel="Tools"),
    instructor: str = typer.Option("", "--instructor", help="Optional instructor prompt text to set on instructor node", rich_help_panel="Tools"),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Print debug info from inside sandbox", rich_help_panel="Debug"),
    guard_empty_diff: bool = typer.Option(False, "--guard-empty-diff/--no-guard-empty-diff", help="Block finish if no staged diff detected (improved runs)", rich_help_panel="Debug"),
    use_default_instructor: bool = typer.Option(False, "--use-default-instructor/--no-use-default-instructor", help="Use the built-in improved instructor prompt", rich_help_panel="Tools"),
    # NOTE: provide any extra arguments if needed
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    if limit and limit > 0:
        instances = instances[:limit]
    print(f"Running on {len(instances)} instances...")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Determine instructor text (CLI string takes precedence, then preset)
        instructor_text = instructor if instructor else (DEFAULT_INSTRUCTOR if use_default_instructor else "")
        futures = {
            executor.submit(
                process_instance,
                instance,
                output_path,
                model_name,
                max_steps,
                backtrack,
                optional_tools,
                instructor_text if instructor_text else None,
                debug,
                guard_empty_diff,
            ): instance[
                "instance_id"
            ]
            for instance in instances
        }
        try:
            process_futures(futures)
        except KeyboardInterrupt:
            print("Cancelling all pending jobs. Press ^C again to exit immediately.")
            for future in futures:
                if not future.running() and not future.done():
                    future.cancel()
            process_futures(futures)


if __name__ == "__main__":
    app()