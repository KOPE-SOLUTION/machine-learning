# Target syntax: Mojo documentation 1.2.0; compiler execution not verified here.
def main():
    var x: List[Float64] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var y: List[Float64] = [-1.0, 0.5, 2.0, 3.5, 5.0]
    var w: Float64 = 0.0
    var b: Float64 = 0.0
    var learning_rate: Float64 = 0.1
    var n = len(x)

    for step in range(300):
        var dw: Float64 = 0.0
        var db: Float64 = 0.0
        for i in range(n):
            var error = w * x[i] + b - y[i]
            dw += 2.0 * error * x[i] / Float64(n)
            db += 2.0 * error / Float64(n)
        w -= learning_rate * dw
        b -= learning_rate * db

    var mse: Float64 = 0.0
    for i in range(n):
        var error = w * x[i] + b - y[i]
        mse += error * error / Float64(n)

    print("w:", w)
    print("b:", b)
    print("training MSE:", mse)

