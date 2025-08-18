import unittest

class TestImports(unittest.TestCase):
    def test_import_prompt_generator(self):
        """
        Tests if the prompt_generator can be imported.
        """
        try:
            from dating_nlp_bot.analyzers import prompt_generator
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import prompt_generator: {e}")

if __name__ == '__main__':
    unittest.main()
