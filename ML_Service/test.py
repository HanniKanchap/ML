import bentoml

model_runner = bentoml.sklearn.get('iris_clf:latest').to_runner()
model_runner.init_local()
print(model_runner.predict.run([[5.9,3.,5.1,1.8]]))