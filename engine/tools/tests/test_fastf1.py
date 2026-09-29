import unittest


class FastF1SmokeTests(unittest.TestCase):
    def test_fastf1_importable(self):
        import fastf1
        self.assertIsNotNone(fastf1.__version__)


if __name__ == "__main__":
    unittest.main()

