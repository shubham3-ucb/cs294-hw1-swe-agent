import sys
import unittest


sys.path.insert(0, "/Users/cusgadmin/Documents/Course Project OpenEvolve/cs294-264-hw-FA25")
from agent import ReactAgent
from response_parser import ResponseParser


class DummyLLM:
    model_name = "dummy"
    def generate(self, prompt: str) -> str:  # pragma: no cover
        return ""


class TestAgentBacktrack(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ReactAgent("test", ResponseParser(), DummyLLM())

    def test_update_instructions_and_backtrack(self):
        # Build a small chain beyond instructor
        a_id = self.agent.add_message("assistant", "step 1")
        b_id = self.agent.add_message("assistant", "step 2")
        # Backtrack to a_id and update instructions
        msg = self.agent.add_instructions_and_backtrack("NEW INSTR", a_id)
        self.assertIn("backtracked to message", msg)
        self.assertEqual(self.agent.current_message_id, a_id)
        # Context should end at a_id (without content from b_id)
        ctx = self.agent.get_context()
        self.assertIn("step 1", ctx)
        self.assertNotIn("step 2", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)


