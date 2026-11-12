"""Tests for deterministic SageMaker registration tokens."""

import re
import unittest

from scripts.register_sagemaker_champion import (
    registration_client_token,
)


class RegistryClientTokenTests(unittest.TestCase):
    def test_token_satisfies_aws_constraints(self):
        token = registration_client_token(
            "retail-global-xgboost-forecasting",
            "a" * 64,
            "b" * 64,
        )

        self.assertEqual(len(token), 36)
        self.assertRegex(token, r"^[a-zA-Z0-9-]+$")

    def test_same_inputs_produce_same_token(self):
        first = registration_client_token(
            "forecasting",
            "a" * 64,
            "b" * 64,
        )

        second = registration_client_token(
            "forecasting",
            "a" * 64,
            "b" * 64,
        )

        self.assertEqual(first, second)

    def test_changed_artifacts_produce_different_tokens(self):
        original = registration_client_token(
            "forecasting",
            "a" * 64,
            "b" * 64,
        )

        changed = registration_client_token(
            "forecasting",
            "a" * 64,
            "c" * 64,
        )

        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
