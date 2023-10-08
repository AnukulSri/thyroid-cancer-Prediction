import bentoml
thyroid_rfc_runner = bentoml.sklearn.get("thyroid_rfc:latest").to_runner()
thyroid_rfc_runner.init_local()
print(thyroid_rfc_runner.predict.run([[17.99,10.38,122.80,1001.0,0.11840]]))
print(thyroid_rfc_runner.predict.run([[11.51,	23.93	,74.52	,403.5,	0.09261]]))
