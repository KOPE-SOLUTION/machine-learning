"""Python reference using the same data, precision, update order, and iterations."""
import numpy as np

x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
y = np.array([-1.0, 0.5, 2.0, 3.5, 5.0], dtype=np.float64)
w, b = 0.0, 0.0
for step in range(300):
    error = w * x + b - y
    dw = 2.0 * np.mean(error * x)
    db = 2.0 * np.mean(error)
    w -= 0.1 * dw
    b -= 0.1 * db
np.testing.assert_allclose([w, b], [3.0, 2.0], atol=1e-8)
print("w:", w)
print("b:", b)
print("training MSE:", np.mean((w * x + b - y) ** 2))

