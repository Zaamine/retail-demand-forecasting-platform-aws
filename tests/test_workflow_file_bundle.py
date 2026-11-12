"""Verify packaging of the cloud forecasting workflow."""

import unittest

from scripts.deploy_forecast_workflow import file_bundle


class WorkflowFileBundleTests(unittest.TestCase):

    def test_required_modules_are_packaged(self):
        files = file_bundle()

        required = [
            "build_cloud_features.py",
            "batch_predict.py",
            "forecasting/build_forecast.py",
            "forecasting/model_inputs.py",
            "forecasting/parity.py",
            "features/horizon.py",
            "features/origin.py",
            "features/target_aligned.py",
        ]

        for name in required:
            with self.subTest(file=name):
                self.assertIn(name, files)

    def test_all_package_files_exist(self):
        for name, path in file_bundle().items():
            with self.subTest(file=name):
                self.assertTrue(path.is_file(), str(path))


if __name__ == "__main__":
    unittest.main()
