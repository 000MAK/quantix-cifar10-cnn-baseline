import unittest
from main import validate_limit

class ArgumentTests(unittest.TestCase):
    def test_positive_value(self): self.assertEqual(validate_limit(10,"epochs"),10)
    def test_reject_zero(self):
        with self.assertRaises(ValueError): validate_limit(0,"batch_size")

if __name__ == "__main__": unittest.main()
