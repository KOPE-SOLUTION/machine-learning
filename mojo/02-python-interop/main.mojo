# Verified with Mojo 1.2.0 on Ubuntu 24.04 / WSL; see mojo/README.md.
from std.python import Python

def main() raises:
    var datasets = Python.import_module("sklearn.datasets")
    var selection = Python.import_module("sklearn.model_selection")
    var pipeline = Python.import_module("sklearn.pipeline")
    var preprocessing = Python.import_module("sklearn.preprocessing")
    var neighbors = Python.import_module("sklearn.neighbors")

    var data = datasets.load_iris(return_X_y=True)
    var split = selection.train_test_split(
        data[0], data[1], test_size=0.2, stratify=data[1], random_state=42
    )
    var model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        neighbors.KNeighborsClassifier(n_neighbors=5)
    )
    _ = model.fit(split[0], split[2])
    print("test accuracy:", model.score(split[1], split[3]))
    print("first five predictions:", model.predict(split[1])[:5])

