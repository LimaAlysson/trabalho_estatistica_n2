from fractions import Fraction
import sympy as sp
from math import log
import numpy as np

#variáveis
t, k = sp.symbols('t k', real=True, positive=True)

#definição da função
f_t = (1/165) * sp.exp(-k*t)

#para ser fdp
integral = sp.integrate(f_t, (t, 0, sp.oo))
valor_k = sp.solve(integral - 1, k)
k_como_fracao = Fraction(float(valor_k[0])).limit_denominator()

print('\nQuestão 02 - Item (a)')
print(f'k = {k_como_fracao} = {float(valor_k[0])}')

####################################

print('\nQuestão 02 - Item (b)')
# P(T > 100)
prob = sp.integrate(f_t, (t, 100, sp.oo)).subs(k, valor_k[0]).evalf()
print(f'P(T > 100) = {prob:.4f} = {prob*100:.2f}%')

####################################

print('\nQuestão 02 - Item (c)')
# E(T)
E_T = sp.integrate(t * f_t, (t, 0, sp.oo)).subs(k, valor_k[0]).evalf()
# E(T²)
E_T2 = sp.integrate(t**2 * f_t, (t, 0, sp.oo)).subs(k, valor_k[0]).evalf()
variancia = E_T2 - E_T**2


print(f'O tempo de vida médio E(T) = {E_T:.4f} horas')
print(f'A variância V(T)= {variancia:.4f} horas²')

####################################

print('\nQuestão 02 - Item (e)')

#Pegando 100000 valores aleatórios entre 0 e 1
n = np.random.rand(100000)
#transformando em array
n = np.array(n)
#distribuição uniforme
t = -165 * np.log(1 - n)
print(t)
