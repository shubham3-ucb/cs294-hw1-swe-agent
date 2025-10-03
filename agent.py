"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect
import time

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm
        self.timestamp: int = int(time.time())

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}
        # Track recently edited files (hints for diff validation)
        self._recent_files: set[str] = set()

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        self.system_message_id = self.add_message("system", "You are a Smart ReAct agent.")
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message("instructor", "")
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE TREE --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        # Determine the new message id (0-based index for list addressing)
        new_message_id: int = len(self.id_to_message)

        # The parent is the current node, unless this is the first (root) message
        parent_id: int | None = self.current_message_id if self.current_message_id != -1 else None

        message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
            "unique_id": new_message_id,
            "parent": parent_id,
            "children": [],
        }

        # Append to storage
        self.id_to_message.append(message)

        # Link from parent to child if applicable
        if parent_id is not None:
            self.id_to_message[parent_id]["children"].append(new_message_id)
        else:
            # First message becomes the root
            self.root_message_id = new_message_id

        # Move the current pointer to the newly created message
        self.current_message_id = new_message_id
        return new_message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        if message_id < 0 or message_id >= len(self.id_to_message):
            raise IndexError("message_id out of range")
        self.id_to_message[message_id]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        if self.current_message_id == -1:
            return ""

        # Walk ancestors from current to root via parent pointers
        path_ids: List[int] = []
        cursor = self.current_message_id
        while cursor is not None and cursor != -1:
            path_ids.append(cursor)
            parent = self.id_to_message[cursor]["parent"]
            cursor = parent if parent is not None else -1

        # Reverse to get root → ... → current ordering
        path_ids.reverse()

        # Concatenate formatted messages
        context_parts: List[str] = []
        for mid in path_ids:
            context_parts.append(self.message_id_to_context(mid))
        return "".join(context_parts)

    # -------------------- HELPER APIS --------------------
    def set_user_prompt(self, user_prompt: str) -> None:
        """Set the user prompt content on the pre-created user node."""
        self.set_message_content(self.user_message_id, user_prompt)

    def get_instructions(self) -> str:
        """Return the current content of the instruction node."""
        return self.id_to_message[self.instructions_message_id]["content"]

    def set_instructions(self, instructions: str) -> None:
        """Set the instruction node content."""
        self.set_message_content(self.instructions_message_id, instructions)

    def save_history(self, file_name: str) -> None:
        """Serialize core agent state to a YAML file for inspection/reproducibility."""
        import yaml
        data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "root_message_id": self.root_message_id,
            "current_message_id": self.current_message_id,
            "messages": self.id_to_message,
        }
        with open(file_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for tool in tools:
            if not callable(tool):
                raise TypeError("All tools must be callable")
            self.function_map[tool.__name__] = tool
        # Nothing else to do here; the system prompt text is dynamically rendered
        # in message_id_to_context using self.function_map and ResponseParser.response_format.
    
    def finish(self, result: str):
        """End the run and return the final summary/result string.

        Note:
            This function does not stage or generate diffs. Before calling it,
            you should stage changes with "git add -A" and verify a non-empty
            staged diff using the "stage_and_diff" tool. The harness will
            collect the staged patch after completion.

        Args:
            result (str): Brief summary of the fix or rationale.

        Returns:
            str: The same result string, which the agent.run() method returns.
        """
        return result 

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # Coerce and validate message id
        try:
            at_message_id = int(at_message_id)
        except Exception:
            raise ValueError("at_message_id must be an integer")
        if at_message_id < 0 or at_message_id >= len(self.id_to_message):
            raise IndexError("at_message_id out of range")

        # Update the instruction node content
        self.set_message_content(self.instructions_message_id, instructions)

        # Move the current pointer to the target node (backtrack)
        self.current_message_id = at_message_id

        return f"Updated instructions and backtracked to message {at_message_id}."

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int, guard_empty_diff: bool = False, debug: bool = False) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        # Respect assignment cap
        step_limit = min(int(max_steps), 100)

        # Set the task on the user node
        self.set_user_prompt(task)
        # Ensure current pointer is at the instruction node for initial context
        self.current_message_id = self.instructions_message_id

        for _ in range(step_limit):
            # Build context and query LLM
            prompt = self.get_context()
            if debug:
                print("[DEBUG] PROMPT (truncated 80 chars):\n" + prompt[:80])
            model_output = self.llm.generate(prompt)
            if debug:
                print("[DEBUG] LLM OUTPUT (truncated 800 chars):\n" + (model_output[:800] if isinstance(model_output, str) else str(model_output)[:800]))
            # Store assistant's raw output
            assistant_id = self.add_message("assistant", model_output)

            # Parse final function call
            try:
                parsed = self.parser.parse(model_output)
            except Exception as e:
                self.add_message("tool", f"Error parsing model output: {e}")
                if debug:
                    print(f"[DEBUG] PARSE ERROR: {e}")
                continue
            fn_name: str = parsed["name"]
            fn_args: Dict[str, Any] = parsed.get("arguments", {})
            if debug:
                print(f"[DEBUG] PARSED CALL: {fn_name} ARGS: {list(fn_args.keys())}")

            if fn_name not in self.function_map:
                # Record the error and continue
                self.add_message("tool", f"Error: unknown function '{fn_name}'")
                continue

            tool_fn = self.function_map[fn_name]

            # Call with best-effort kwargs matching the function signature
            try:
                sig = inspect.signature(tool_fn)
                accepted = {
                    name: fn_args[name]
                    for name in sig.parameters.keys()
                    if name in fn_args
                }
                # Record file edit hints for certain tools
                if fn_name in {"replace_in_file", "show_file", "show_file_range"}:
                    file_arg = accepted.get("file_path")
                    if isinstance(file_arg, str) and file_arg:
                        self._recent_files.add(file_arg)
                # Guards before executing certain tools
                if fn_name == "replace_in_file":
                    file_arg = accepted.get("file_path")
                    from_line = accepted.get("from_line")
                    to_line = accepted.get("to_line")
                    # Require prior inspection of the file to reduce irrelevant edits
                    if isinstance(file_arg, str) and file_arg and file_arg not in self._recent_files:
                        raise ValueError(
                            "Edit blocked: inspect the file first with show_file/show_file_range."
                        )
                    # Discourage very large edits; encourage surgical changes
                    try:
                        f = int(from_line)
                        t = int(to_line)
                        if t - f > 120:
                            raise ValueError("Edit blocked: requested range too large (>120 lines).")
                    except Exception:
                        # If coercion fails, let the tool handle validation
                        pass

                if debug:
                    print(f"[DEBUG] CALL TOOL: {fn_name} ARGS: {accepted}")
                result = tool_fn(**accepted)
            except Exception as e:
                # Attach error message and continue
                self.add_message("tool", f"Error executing {fn_name}: {e}")
                if debug:
                    print(f"[DEBUG] TOOL ERROR: {fn_name}: {e}")
                continue

            # Append tool result
            tool_output = str(result)
            if debug:
                print(f"[DEBUG] TOOL OUTPUT (truncated 400 chars):\n{tool_output[:400]}")
            self.add_message("tool", tool_output)

            # If finish, possibly guard against empty diffs
            if tool_fn is self.finish or fn_name == "finish":
                if guard_empty_diff:
                    # Heuristic: ensure a staged diff was produced recently
                    if not self._recent_tool_output_has_diff():
                        self.add_message(
                            "tool",
                            "Guard blocked finish: no non-empty staged diff detected. Continue editing and call stage_and_diff before finish.",
                        )
                        continue
                return tool_output

        # Step limit reached
        raise RuntimeError("Step limit reached without calling finish")

    def _recent_tool_output_has_diff(self) -> bool:
        """Scan recent tool messages for a unified diff marker."""
        count = 0
        for msg in reversed(self.id_to_message):
            if msg["role"] != "tool":
                continue
            content = msg.get("content", "") or ""
            if "diff --git" in content or content.startswith("+++ ") or content.startswith("--- "):
                # If we have recent file hints, require at least one to appear in the diff
                if self._recent_files:
                    for f in self._recent_files:
                        # Check by basename to be resilient to path prefixes
                        import os
                        base = os.path.basename(f)
                        if base and base in content:
                            return True
                    # No hinted file found; keep scanning other tool outputs
                else:
                    return True
            count += 1
            if count >= 10:
                break
        return False

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()