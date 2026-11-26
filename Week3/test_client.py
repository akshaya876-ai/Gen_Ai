
import unittest
from unittest.mock import Mock

from llm_client import LLMClient, LLMResult


class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.mock_provider = Mock()
        self.mock_logger = Mock()

        self.client = LLMClient(
            self.mock_provider,
            self.mock_logger
        )

    def test_successful_generation(self):
        self.mock_provider.generate.return_value = LLMResult(
            text="Hello!",
            input_tokens=5,
            output_tokens=3
        )

        result = self.client.generate("Say hello")

        self.assertEqual(result.text, "Hello!")
        self.assertEqual(result.input_tokens, 5)
        self.assertEqual(result.output_tokens, 3)

        self.mock_logger.log_call.assert_called_once()

    def test_failed_generation(self):
        self.mock_provider.generate.side_effect = Exception("API error")

        with self.assertRaises(Exception):
            self.client.generate("Test prompt")

        self.assertEqual(self.mock_provider.generate.call_count, 3)
        self.assertEqual(self.mock_logger.log_call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
