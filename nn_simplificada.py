import numpy as np
import matplotlib.pyplot as plt

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)

class NN:
    def __init__(self):
        np.random.seed(42)
        self.input_size = 1
        self.hidden_size = 3
        self.output_size = 1

        # Inicialização dos pesos e bias
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

        self.error_history = []
        self.weight_history = []

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)  # Sigmoid na camada oculta
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = linear(self.z2)   # Linear na camada de saída
        return self.a2

    def backward(self, x, y, lr=0.1):
        m = len(x)
        output = self.a2
        error = output - y
        self.error_history.append(np.mean(error**2))  # Armazena MSE

        # Backpropagation
        dz2 = error * linear_deriv(output)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * sigmoid_deriv(self.a1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Atualização dos pesos
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.weight_history.append((
            self.W1.copy(),
            self.W2.copy()
        ))

    def train(self, x, y, epochs=100, lr=0.1):
        plt.figure(figsize=(12, 5))
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            
            if epoch % 10 == 0 or epoch == epochs-1:
                self.plot_progress(epoch, x, y)
                
    def plot_progress(self, epoch, x, y):
        plt.clf()
        
        # Gráfico 1: Dados e previsões
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, label='Dados reais', color='blue')
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        plt.plot(x_range, self.forward(x_range), 'r-', label='Previsão')
        plt.title(f'Época {epoch}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        
        # Gráfico 2: Evolução do erro
        plt.subplot(1, 2, 2)
        plt.plot(self.error_history, 'g-')
        plt.title('Erro (MSE) durante o treino')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)

    def predict(self, x):
        return self.forward(x)

# Dados de treinamento (y = x)
x_train = np.array([[0], [0.2], [0.4], [0.6], [0.8], [1.0]])
y_train = x_train.copy()

# Treinamento
model = NN()
plt.ion()
model.train(x_train, y_train, epochs=500, lr=0.01)
plt.ioff()

# Teste
x_test = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
predictions = model.predict(x_test)

print("\nTeste:")
for x, pred in zip(x_test, predictions):
    print(f"x = {x[0]:.1f} → Previsto: {pred[0]:.4f} (Desejado: {x[0]:.1f})")