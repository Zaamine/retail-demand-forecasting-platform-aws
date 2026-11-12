"""Tests for SageMaker Model Package Group existence checks."""

import unittest
from unittest.mock import Mock

from botocore.exceptions import ClientError

from scripts.register_sagemaker_champion import describe_existing_group


GROUP = "retail-global-xgboost-forecasting"


def aws_error(code: str, message: str) -> ClientError:
    """Build an AWS ClientError with a realistic response structure."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "DescribeModelPackageGroup",
    )


class RegistryGroupLookupTests(unittest.TestCase):
    def test_existing_group(self):
        client = Mock()
        client.describe_model_package_group.return_value = {
            "ModelPackageGroupName": GROUP,
            "ModelPackageGroupStatus": "Completed",
        }

        self.assertTrue(describe_existing_group(client, GROUP))

    def test_missing_group_validation_exception(self):
        client = Mock()
        client.describe_model_package_group.side_effect = aws_error(
            "ValidationException",
            f"ModelPackageGroup arn:aws:sagemaker:eu-west-1:"
            f"123456789012:model-package-group/{GROUP} does not exist.",
        )

        self.assertFalse(describe_existing_group(client, GROUP))

    def test_missing_group_resource_not_found(self):
        client = Mock()
        client.describe_model_package_group.side_effect = aws_error(
            "ResourceNotFound",
            "Resource not found.",
        )

        self.assertFalse(describe_existing_group(client, GROUP))

    def test_access_denied_is_not_silenced(self):
        client = Mock()
        client.describe_model_package_group.side_effect = aws_error(
            "AccessDeniedException",
            "Not authorized to describe this model package group.",
        )

        with self.assertRaises(ClientError):
            describe_existing_group(client, GROUP)

    def test_unrelated_validation_error_is_not_silenced(self):
        client = Mock()
        client.describe_model_package_group.side_effect = aws_error(
            "ValidationException",
            "Invalid model package group name.",
        )

        with self.assertRaises(ClientError):
            describe_existing_group(client, GROUP)


if __name__ == "__main__":
    unittest.main()
