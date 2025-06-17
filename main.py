import numpy as np
import matplotlib.pyplot as plt
from nn_simplificada import NN  # Assumindo que esta é sua classe

# 1. Criar dataset com ruído (mantendo intervalo controlado)
np.random.seed(42)
x_train = np.linspace(0, 10, 50).reshape(-1, 1)
noise = np.random.uniform(-1, 1, x_train.shape)
y_train = 4 * x_train + 5 + noise  # y = 4x + 5 + ruído

# 2. Normalizar os dados (essencial para sigmoide!)
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std

# 3. Treinar a rede COM dados normalizados
nn = NN()
plt.ion()
nn.train(x_train_norm, y_train_norm, epochs=100, lr=0.1)
plt.ioff()

# 4. Testar em intervalo similar ao de treino (0-10)
x_test = np.linspace(0, 10, 100).reshape(-1, 1)
x_test_norm = (x_test - x_mean) / x_std

# Fazer previsões e desnormalizar
y_pred_norm = nn.forward(x_test_norm)
y_pred = y_pred_norm * y_std + y_mean
y_true = 4 * x_test + 5

# 5. Plotar comparando com dados reais
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Dados de Treino', alpha=0.6)
plt.plot(x_test, y_true, 'g-', label='Relação Real (y=4x+5)')
plt.plot(x_test, y_pred, 'r--', label='Previsões da Rede')
plt.title('Comparação: Previsões vs Real (Dados Normalizados)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()