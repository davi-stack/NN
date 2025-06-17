import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=5):
        np.random.seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

        # Inicialização dos pesos
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2./self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2./self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.train_loss = []
        self.val_loss = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = linear(self.z2)
        return self.a2

    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        
        # Cálculo dos gradientes
        output_error = self.a2 - y
        dW2 = np.dot(self.a1.T, output_error) / m
        db2 = np.sum(output_error, axis=0, keepdims=True) / m
        
        hidden_error = np.dot(output_error, self.W2.T) * sigmoid_deriv(self.a1)
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m

        # Atualização dos pesos
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            # Forward e backward pass
            predictions = self.forward(X_train)
            self.backward(X_train, y_train, lr)
            
            # Cálculo do erro
            train_mse = np.mean((predictions - y_train)**2)
            val_predictions = self.forward(X_val)
            val_mse = np.mean((val_predictions - y_val)**2)
            
            self.train_loss.append(train_mse)
            self.val_loss.append(val_mse)
            
            if epoch % 100 == 0:
                print(f"Época {epoch}: Train MSE = {train_mse:.4f}, Val MSE = {val_mse:.4f}")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('Curva de Aprendizado')
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, X):
        return self.forward(X)

data = pd.read_csv('dataset/dados.csv')

# Selecionar features (usando as 3 mais relevantes)
X = data[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']].values
y = data['Y house price of unit area'].values.reshape(-1, 1)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Criar e treinar a rede
input_size = X_train_scaled.shape[1]
nn = NeuralNetwork(input_size=input_size, hidden_size=10)
nn.train(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, epochs=2000, lr=0.01)

# Plotar a curva de aprendizado
nn.plot_loss()

# Avaliar no conjunto de teste
predictions_scaled = nn.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# Métricas de avaliação
mse = np.mean((predictions - y_test)**2)
mae = np.mean(np.abs(predictions - y_test))
r2 = 1 - np.sum((y_test - predictions)**2) / np.sum((y_test - np.mean(y_test))**2)


print("\nMétricas de Avaliação:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Visualizar previsões vs reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, c='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs Valores Reais')
plt.grid()
plt.show()