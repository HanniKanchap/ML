from sklearn.datasets import load_iris
from sklearn.svm import SVC
import bentoml

iris = load_iris()
X,y = iris.data , iris.target

model = SVC(gamma='scale')
model.fit(X,y)

saved_model = bentoml.sklearn.save_model('iris_clf',model)
print(f"Model saved : {saved_model}")

## iris_clf:bweezcabaoetuacq