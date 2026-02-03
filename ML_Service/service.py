import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load model runner
model_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Define service
svc = bentoml.Service("iris_classifier", runners=[model_runner])

# Define API
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    return model_runner.predict.run(input_series)