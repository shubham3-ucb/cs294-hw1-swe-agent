import sys
import unittest


sys.path.insert(0, "/Users/cusgadmin/Documents/Course Project OpenEvolve/cs294-264-hw-FA25")
from response_parser import ResponseParser


class TestResponseParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ResponseParser()

    def test_simple_one_arg(self):
        text = "\n".join(
            [
                "Let me think...",
                "----BEGIN_FUNCTION_CALL----",
                "run_bash_cmd",
                "----ARG----",
                "command",
                "ls -la",
                "----END_FUNCTION_CALL----",
            ]
        )
        out = self.parser.parse(text)
        self.assertEqual(out["thought"], "Let me think...")
        self.assertEqual(out["name"], "run_bash_cmd")
        self.assertEqual(out["arguments"], {"command": "ls -la"})

    def test_multiple_args_and_multiline_value(self):
        text = "\n".join(
            [
                "reasoning",
                "----BEGIN_FUNCTION_CALL----",
                "replace_in_file",
                "----ARG----",
                "file_path",
                "app/main.py",
                "----ARG----",
                "from_line",
                "10",
                "----ARG----",
                "to_line",
                "20",
                "----ARG----",
                "content",
                "line1",
                "line2",
                "line3",
                "----END_FUNCTION_CALL----",
            ]
        )
        out = self.parser.parse(text)
        self.assertEqual(out["name"], "replace_in_file")
        self.assertEqual(out["arguments"]["file_path"], "app/main.py")
        self.assertEqual(out["arguments"]["to_line"], "20")
        self.assertEqual(out["arguments"]["content"].splitlines()[-1], "line3")

    def test_missing_end_marker(self):
        with self.assertRaises(ValueError):
            self.parser.parse("no end\n----BEGIN_FUNCTION_CALL----\nfn")

    def test_missing_begin_marker(self):
        with self.assertRaises(ValueError):
            self.parser.parse("thought only\n----END_FUNCTION_CALL----")

    def test_missing_function_name(self):
        text = "\n".join(
            [
                "reasoning",
                "----BEGIN_FUNCTION_CALL----",
                "",
                "----ARG----",
                "arg",
                "val",
                "----END_FUNCTION_CALL----",
            ]
        )
        with self.assertRaises(ValueError):
            self.parser.parse(text)


if __name__ == "__main__":
    unittest.main(verbosity=2)


