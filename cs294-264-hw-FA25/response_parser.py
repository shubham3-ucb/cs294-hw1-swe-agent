class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
arg2_value (can be multiline)
...
{END_CALL}
"""

    def parse(self, text: str) -> dict:
        """
        Parse and extract the final function call embedded in `text`.

        The parser is resilient to delimiter-like content that may appear earlier in
        the reasoning by locating the last occurrence of the END marker and then the
        corresponding BEGIN marker using rfind.

        Returns a dictionary with keys:
        - "thought": str, free-form reasoning text preceding the function call
        - "name": str, the function name
        - "arguments": dict[str, str], argument name to (possibly multiline) value

        Raises ValueError if the text is malformed or delimiters are missing.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            raise ValueError("Missing END_FUNCTION_CALL delimiter")

        begin_idx = text.rfind(self.BEGIN_CALL, 0, end_idx)
        if begin_idx == -1:
            raise ValueError("Missing BEGIN_FUNCTION_CALL delimiter")

        thought = text[:begin_idx].strip()
        inner = text[begin_idx + len(self.BEGIN_CALL): end_idx]

        # Split by ARG separator; the first block is the function name
        parts = inner.split(self.ARG_SEP)
        if not parts:
            raise ValueError("Malformed function call: empty body")

        func_name = parts[0].strip()
        if not func_name:
            raise ValueError("Malformed function call: missing function name")

        arguments: dict[str, str] = {}
        for block in parts[1:]:
            # Normalize spacing but preserve value newlines
            segment = block.lstrip("\n")
            if not segment.strip():
                # Allow empty blocks (robustness), skip
                continue
            newline_idx = segment.find("\n")
            if newline_idx == -1:
                raise ValueError("Malformed argument block: expected name and value")
            arg_name = segment[:newline_idx].strip()
            arg_value = segment[newline_idx + 1:]
            # Trim only surrounding newlines and spaces at ends; preserve internal formatting
            arg_value = arg_value.strip()
            if not arg_name:
                raise ValueError("Malformed argument block: missing argument name")
            arguments[arg_name] = arg_value

        return {"thought": thought, "name": func_name, "arguments": arguments}
