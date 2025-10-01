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
            patch_output = self.env.execute("git add -A && git diff --cached")
            if patch_output.strip():
                return patch_output
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"
    
    # -------------------- TODO(student): add more functions here if you want --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace the content of the file from the given line to the given line with the given content
        """
        if from_line < 1 or to_line < from_line:
            raise ValueError("Invalid line range")
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        # Use an inline Python script inside the target environment to avoid sed/awk edge cases
        cmd = f"""
python - << 'PY'
import base64
import io
import sys

p = {file_path!r}
start = {from_line} - 1  # 0-based inclusive
end = {to_line}          # 0-based exclusive
replacement = base64.b64decode({b64!r}).decode('utf-8').splitlines(True)

with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

if start > len(lines):
    # If start beyond EOF, just append replacement at EOF
    start = len(lines)
if end > len(lines):
    end = len(lines)

new_lines = lines[:start] + replacement + lines[end:]
with open(p, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Replaced lines {start+1}-{end} in {p}")
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