from collections import Counter
import numpy as np

# Definir classe KNN para definir como será o treinamento
class KNN:
    # Construtor
    def __init__(self, k=3):
        self.k = k

    # Setar o conjunto de treinamento e as classes esperadas
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Realizar treinamento
    def predict(self, X):
        y_pred = []
        # Calcular a distância de cada ponto para todos os outros
        for x in X:
            k_idx = self.k_closest_points(x)
            y_pred.append(self.classify(k_idx))
        
        return np.array(y_pred)

    # Para cada ponto, armazenar em um array a distancia dele para todos os outros
    def k_closest_points(self, x):
        distances = [KNN.euclidian_distance(x, x_train) for x_train in self.X_train]
        # Pegar o k primeiros vizinhos
        k_idx = np.argsort(distances)[: self.k]
        return k_idx

    # Classificação
    def classify(self, k_idx):
        # Extrair os labels dos k primeiros vizinhos
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # Majority Voting - escolher dos k primeiros vizinhos a classificação
        # Retornar a label mais comum
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    # Medida de similaridade (usado a distância euclidiana)
    @staticmethod
    def euclidian_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Acurácia
    @staticmethod
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy