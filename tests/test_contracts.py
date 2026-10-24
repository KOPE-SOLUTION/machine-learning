"""Behavioral checks for leakage, math, and saved-model input contracts."""
import importlib.util
from io import BytesIO
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]


def load_lesson(slug):
    spec = importlib.util.spec_from_file_location(slug, ROOT / "lessons" / slug / "lesson.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LearningContracts(unittest.TestCase):
    def test_gradient_descent_matches_known_line(self):
        module = load_lesson("15-gradient-descent")
        x = np.linspace(-1, 1, 101)
        w, b, history = module.fit_gradient_descent(x, 3 * x + 2)
        np.testing.assert_allclose([w, b], [3, 2], atol=1e-6)
        self.assertLess(history[-1], history[0])

    def test_gradient_descent_rejects_mismatched_vectors(self):
        module = load_lesson("15-gradient-descent")
        with self.assertRaises(ValueError):
            module.fit_gradient_descent([1, 2], [3])

    def test_gradient_descent_rejects_invalid_settings(self):
        module = load_lesson("15-gradient-descent")
        for kwargs in ({"learning_rate": np.nan}, {"learning_rate": np.inf},
                       {"learning_rate": 0}, {"steps": 1.5}, {"steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.fit_gradient_descent([0, 1], [2, 5], **kwargs)

    def test_gradient_descent_stops_when_loss_overflows(self):
        module = load_lesson("15-gradient-descent")
        with self.assertRaisesRegex(FloatingPointError, "reduce learning_rate"):
            module.fit_gradient_descent([1, 2], [3, 5], learning_rate=1e200)

    def test_pixel_contract_rejects_bad_inputs(self):
        module = load_lesson("16-digits-project")
        for bad in (np.zeros(64), np.zeros((1, 63)), np.full((1, 64), np.nan),
                    np.full((1, 64), 17), np.zeros((0, 64))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                module.validate_pixels(bad)

    def test_scaler_uses_train_statistics_and_roundtrip_preserves_predictions(self):
        module = load_lesson("16-digits-project")
        X, y = load_digits(return_X_y=True)
        train = np.arange(300)
        held_out = np.arange(300, 340)
        model = module.build_search().estimator.fit(X[train], y[train])
        np.testing.assert_allclose(model.named_steps["scale"].mean_, X[train].mean(axis=0))
        pred = model.predict(module.validate_pixels(X[held_out]))
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        np.testing.assert_array_equal(pred, joblib.load(buffer).predict(X[held_out]))


if __name__ == "__main__":
    unittest.main()
