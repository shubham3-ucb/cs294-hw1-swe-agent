import sys
import unittest


sys.path.insert(0, "/Users/cusgadmin/Documents/Course Project OpenEvolve/cs294-264-hw-FA25")
from agent import ReactAgent
from response_parser import ResponseParser


class DummyLLM:
    model_name = "dummy"
    def generate(self, prompt: str) -> str:  # pragma: no cover
        return ""


class TestAgentMessageTree(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ReactAgent("test", ResponseParser(), DummyLLM())

    def test_tree_structure_and_context_order(self):
        # After __init__, system, user, and instructor messages exist
        self.assertGreaterEqual(len(self.agent.id_to_message), 3)
        self.assertEqual(self.agent.id_to_message[self.agent.root_message_id]["role"], "system")

        # Add an assistant message and a tool message, then verify context order ends with latest
        a_id = self.agent.add_message("assistant", "step 1")
        t_id = self.agent.add_message("tool", "ok")
        ctx = self.agent.get_context()
        self.assertIn("|MESSAGE(role=\"system\"", ctx)
        self.assertIn("|MESSAGE(role=\"user\"", ctx)
        self.assertIn("|MESSAGE(role=\"instructor\"", ctx)
        self.assertIn("|MESSAGE(role=\"assistant\"", ctx)
        self.assertIn("|MESSAGE(role=\"tool\"", ctx)
        self.assertTrue(ctx.rfind("|MESSAGE(role=\"tool\"") > ctx.rfind("|MESSAGE(role=\"assistant\""))


if __name__ == "__main__":
    unittest.main(verbosity=2)


