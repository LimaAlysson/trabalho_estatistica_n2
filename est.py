import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress




############ 4ª QUESTÃO 
######################### ITEM A

#valores
X = [77,68,65,73,84,76,82,78,77,79,80,90,84,91,84,80,87,97,101,105,100,101,96,100,97,101,95,100,99,101]
Y = [43,43,45,45,45,40,49,53,53,54,53,56,56,58,64,65,65,65,67,68,68,71,71,71,73,76,76,78,79,80]

X = np.array(X)
Y = np.array(Y)

#calculo das médias
media_X = np.mean(X)
media_Y = np.mean(Y)

#diferenças para a média
dif_X = X - media_X
dif_Y = Y - media_Y

#produto das diferenças
prod_diferencas = [dx * dy for dx, dy in zip(dif_X, dif_Y)]

#quadrado das diferenças
quadrado_dif_X = [dx**2 for dx in dif_X]
quadrado_dif_Y = [dy**2 for dy in dif_Y]

r = sum(prod_diferencas) / np.sqrt(sum(quadrado_dif_X)*sum(quadrado_dif_Y))

print(f"\nQuestão 04 - Item (a) \nCoeficiente de Correlação de Pearson (r): {r:.4f}\n")

######################### ITEM B
# Y = a + bX
# a = coeficiente linear
# b = coeficiente angular

#Regressão Linear
slope, intercept, _, _, _ = linregress(X, Y)
#Equação da reta
Y_pred_linear = intercept + slope * X 

#Regressão Quadrática
coef_quad = np.polyfit(X, Y, 2) # Ajustando um polinômio de grau 2
Y_pred_quad = np.polyval(coef_quad, X) #Calcula valores preditos

# Representação gráfica
plt.scatter(X, Y, color='black', label="Dados")
plt.plot(X, Y_pred_linear, color='blue', label="Regressão Linear Simples")
plt.plot(X, Y_pred_quad, color='red', linestyle="dashed", label="Regressão Quadrática")

#configs do gráfico
plt.xlabel("Idade (anos)")
plt.ylabel("Massa Muscular")
plt.title("Relação entre a massa muscular e a idade (em anos) de uma amostra de 30 indivíduos ")
plt.legend()
plt.grid()

plt.show() #mostrar o gráfico
print(f"Questão 04 - Item (b)\nGráfico - Ajuste de Regressão Linear Simples e Quadrática\n")

######################### ITEM C

# Coeficiente de Determinação R²  Linear
SS_res_linear = np.sum((Y - Y_pred_linear) ** 2)
SS_tot = np.sum((Y - np.mean(Y)) ** 2)
R2_linear = 1 - (SS_res_linear / SS_tot)

# Coeficiente de Determinação Ajustado R² Linear
n = len(X)
p_linear = 1
R2_ajustado_linear = 1 - ((1 - R2_linear) * (n -1)) / (n - p_linear - 1)

# Coeficiente de Determinação R² Quadrático
SS_res_quad = np.sum((Y - Y_pred_quad) ** 2)
R2_quad = 1 - (SS_res_quad / SS_tot)

# Coeficiente de Determinação Ajustatado R² Quadrático
p_quad = 2 # é o nº de parâmetros no modelo quadrático (incluindo o intercepto)
R2_ajustado_quad = 1 - ((1 - R2_quad) * (n - 1)) / (n - p_quad - 1)

print("Questão 04 - Item (c)")
print(f"Coeficiente de determinação Linear: {R2_linear:.4f}")
print(f"Coeficiente de determinação Linear ajustado: {R2_ajustado_linear:.4f}")
print(f"Coeficiente de determinação quadrático: {R2_quad:.4f}")
print(f"Coeficiente de determinação quadrático ajustado: {R2_ajustado_quad:.4f}")

######################### ITEM D




