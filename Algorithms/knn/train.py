# Definir a main para instanciar a base de dados e executar o treinamento
from sklearn import datasets
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from knn import KNN

# Definir a base de dados, usaremos a do sklearn de flores
# Não será necessário fazer tratamento dos dados
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Mostrar a base de dados utilizando o Pandas
df = pd.DataFrame(X, columns = iris.feature_names)
display(df)

# Divir a base em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)

# Chamar o KNN
knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(predictions)

# Testar acurácia do treinamento
# Forma 1
print(f"KNN classification accuracy: {knn.accuracy(y_test, predictions)}\n")

# Forma 2
clr = classification_report(y_test, predictions, target_names=iris.target_names)
print(clr)