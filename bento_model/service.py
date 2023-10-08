import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

thyroid_rfc_runner = bentoml.sklearn.get('thyroid_rfc:latest').to_runner()

rfc = bentoml.Service('thyroid_rfc',runners=[thyroid_rfc_runner])

@rfc.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_series:np.ndarray) -> np.ndarray:
   result = thyroid_rfc_runner.predict.run(input_series)
   return result