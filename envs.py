from utils import get_sb_environment
import subprocess
import base64

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Empty command")
        try:
            output = self.env.execute(command)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
        return output
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            raw = self.env.execute("git add -A && git diff --cached")
            # Normalize possible return types from the environment
            if isinstance(raw, dict):
                # mini-swe-agent may return {'output': str, 'returncode': int}
                output = raw.get("output", "")
                stdout = raw.get("stdout", "")
                stderr = raw.get("stderr", "")
                patch_output = output or stdout or stderr
            else:
                patch_output = raw

            if isinstance(patch_output, bytes):
                patch_output = patch_output.decode("utf-8", errors="replace")

            text = str(patch_output)
            if text.strip():
                return text
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"
    
    # -------------------- TODO(student): add more functions here if you want --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace a line range with the given content, preserving scope indentation.

        Guards:
        - Coerces from/to to integers and validates range
        - Converts literal "\\t" sequences to 4 spaces
        - Normalizes newlines and re-indents the inserted block to match the
          surrounding file scope so methods stay inside their class/function
        """
        # Coerce line numbers from potential string inputs
        try:
            from_line = int(from_line)
            to_line = int(to_line)
        except Exception:
            raise ValueError("from_line/to_line must be integers")

        if from_line < 1 or to_line < from_line:
            raise ValueError("Invalid line range")
        # Basic sanitization: normalize newlines and replace both literal "\\t" and actual tabs with spaces
        sanitized = (
            content.replace("\r\n", "\n")
                   .replace("\r", "\n")
                   .replace("\\t", "    ")
                   .replace("\t", "    ")
        )
        b64 = base64.b64encode(sanitized.encode("utf-8")).decode("ascii")
        # Use an inline Python script inside the target environment to avoid sed/awk edge cases
        cmd = f"""
python - << 'PY'
import base64
import io
import sys
import re

p = {file_path!r}
start = {from_line} - 1  # 0-based inclusive
end = {to_line}          # 0-based exclusive
raw = base64.b64decode({b64!r}).decode('utf-8')
replacement = raw.splitlines(True)

with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

if start > len(lines):
    # If start beyond EOF, just append replacement at EOF
    start = len(lines)
if end > len(lines):
    end = len(lines)

# Detect original indentation prefix from target slice or nearby context
def leading_ws(s: str) -> str:
    m = re.match(r'^[\t ]*', s)
    return m.group(0) if m else ''

orig_prefix = ''
for probe in lines[start:end]:
    if probe.strip():
        orig_prefix = leading_ws(probe)
        break
if not orig_prefix and start > 0:
    # Look upwards a bit for context
    for probe in reversed(lines[max(0, start-20):start]):
        if probe.strip():
            orig_prefix = leading_ws(probe)
            break

# Compute common leading indent in replacement to preserve relative indent
non_empty = [r for r in replacement if r.strip()]
common_ws = None
for r in non_empty:
    ws = leading_ws(r)
    if common_ws is None or len(ws) < len(common_ws):
        common_ws = ws
if common_ws is None:
    common_ws = ''

# If replacement appears to introduce a top-level class/def (after removing
# its own common indent), avoid inheriting an indented context prefix which
# would incorrectly nest the definition inside another scope.
introduces_toplevel_symbol = False
for r in non_empty:
    core = r[len(common_ws):] if r.startswith(common_ws) else r.lstrip('\t ')
    if core.lstrip().startswith(('class ', 'def ')) and core == core.lstrip():
        introduces_toplevel_symbol = True
        break
if introduces_toplevel_symbol and orig_prefix:
    orig_prefix = ''

rebuilt = []
for r in replacement:
    if r.strip():
        if r.startswith(common_ws):
            core = r[len(common_ws):]
        else:
            core = r.lstrip('\t ')
        rebuilt.append(orig_prefix + core)
    else:
        # Preserve blank line as-is
        rebuilt.append(r)

replacement = rebuilt

new_lines = lines[:start] + replacement + lines[end:]
with open(p, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Replaced lines %d-%d in %s" % (start+1, end, p))
PY
"""
        return self.env.execute(cmd)
    
    def show_file(self, file_path: str) -> str:
        """
        [Optional]Show the content of the file
        """
        # Prefer cat -n for portability
        cmd = f"cat -n {file_path}"
        return self.env.execute(cmd)

    def stage_and_diff(self) -> str:
        """
        [Optional] Stage all changes and return the current cached diff.

        Returns a unified diff string (may be empty if no changes staged).
        """
        try:
            raw = self.env.execute("git add -A && git diff --cached")
            if isinstance(raw, dict):
                output = raw.get("output", "")
                stdout = raw.get("stdout", "")
                stderr = raw.get("stderr", "")
                out = output or stdout or stderr
            else:
                out = raw
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            return str(out)
        except Exception as e:
            raise ValueError(f"stage_and_diff failed: {e}")

    def show_file_range(self, file_path: str, from_line: int, to_line: int) -> str:
        """
        [Optional] Show a line range from a file (inclusive), with line numbers.
        """
        try:
            from_line = int(from_line)
            to_line = int(to_line)
        except Exception:
            raise ValueError("from_line/to_line must be integers")
        if from_line < 1 or to_line < from_line:
            raise ValueError("Invalid line range")
        cmd = f"sed -n '{from_line},{to_line}p' {file_path} | nl -ba -v {from_line}"
        return self.env.execute(cmd)

    def grep_repo(self, pattern: str) -> str:
        """
        [Optional] Search recursively for a pattern in the repository, excluding .git.
        Returns filename:line:match lines.
        """
        if not pattern or not isinstance(pattern, str):
            raise ValueError("pattern must be a non-empty string")
        cmd = f"grep -RIn --exclude-dir=.git --line-number {pattern!r} /testbed || true"
        return self.env.execute(cmd)

    def run_common_tests(self) -> str:
        """
        [Optional] Try a sequence of common test commands; return outputs.
        Does not raise on failure; returns accumulated logs.
        """
        commands = [
            "pytest -q || true",
            "pytest -q /testbed || true",
            "python -m pytest -q || true",
            "python -m pytest -q /testbed || true",
            "tox -q || true",
            "python runtests.py -q || true",
            "python setup.py test || true",
        ]
        logs = []
        for c in commands:
            try:
                out = self.env.execute(c)
            except Exception as e:
                out = str(e)
            logs.append(f"$ {c}\n{out}")
        return "\n\n".join(logs)

    def syntax_check(self) -> str:
        """
        [Optional] Quickly check Python syntax across the repo.

        Runs compileall and returns output without raising on failure.
        """
        try:
            return self.env.execute("python -m compileall -q /testbed || true")
        except Exception as e:
            return str(e)

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output