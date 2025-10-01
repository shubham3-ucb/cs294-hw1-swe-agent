import sys
import unittest


sys.path.insert(0, "/Users/cusgadmin/Documents/Course Project OpenEvolve/cs294-264-hw-FA25")
from agent import ReactAgent
from response_parser import ResponseParser


class FinishLLM:
    model_name = "dummy"
    def __init__(self, stop_token: str):
        self.stop_token = stop_token
    def generate(self, prompt: str) -> str:  # pragma: no cover
        return (
            "thinking...\n"
            "----BEGIN_FUNCTION_CALL----\n"
            "finish\n"
            "----ARG----\n"
            "result\n"
            "done\n"
            "----END_FUNCTION_CALL----"
        )


class TestAgentRun(unittest.TestCase):
    def test_run_finishes(self):
        agent = ReactAgent("test", ResponseParser(), FinishLLM(ResponseParser.END_CALL))
        out = agent.run("echo task", max_steps=3)
        self.assertEqual(out, "done")


if __name__ == "__main__":
    unittest.main(verbosity=2)


