# algebra




# Informacion sobre listas, dic, duplas,vectores
from numpy import matmul
from numpy import trace
from numpy import trace, array, matmul, power, zeros, sum
from numpy import sum
from numpy import transpose
from numpy import array, cos, sin, cross, pi, linalg
from numpy.linalg import norm
from numpy import dot
from math import cos, sin
from numpy import array
import numpy as np
import sympy as sp
from numpy.linalg import norm, det
from math import e, pi
import math

# Vector
V = array([1, 2, 3])
W = array([1, 2, 5])
K = array([6, 5, 5])
Matrix = array([V, W, K])
print(Matrix)
T = transpose(Matrix)
print(T)
# print(Matrix)
# # Módulo
print(norm(V))
# Determinnante
det(Matrix)
print(det(Matrix))
Lista = [2, 3, 4, 5, 6, 2, 1, -3,]
# print(L)


# # Quitar el más pequeño y más grande
# L.remove(min(L))
# print(Lista)
# L.remove(max(L))
# print(Lista)

# # Identificar elementos en listas y su intervalo y suma de sus valores
L = [2, 3, 4, 5, 6, 2, 1, -3,]
(a, b) = (-3, 10)
n = 0
s = 0
for i in L:
    if i >= a and i <= b:
        n += 1
        s += i
        print(i)
print(n)
print(s)


# Operaciones basicas
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([9, 5, 5])
# # Calcular el producto vectorial


def cross_product(a, b):
    k = np.cross(a, b)
    print(k)


def doublecross_product(a, b, c):  # a^b^c
    k = np.cross(a, b)
    print(k)
    final = np.cross(k, c)
    print(final)


def cross(a, b):
    cd = [a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]]

    return cd


# Calcular trazas y multiplicaciondes de matrices

# Primera matriz
def Vandermonde(N):
    A = np.array([[(i/float(N))**(j-1) for j in range(1, N+1)]
                  for i in range(1, N+1)])
    return A


# print(Vandermonde(4))

# 3Calcular trazas

print(trace(Matrix))


def traZa(N):
    trace = 0

    Matr = Matrix  # Se puede escoger cualquier matriz
    for i in range(N):
        trace += Matr[i, i]
    return trace
    return Vandermonde(N)


# print(traza(Matrix))


def TRAZA(Matriz):
    TRAZA = 0
    for i in range(Matriz.shape[0]):
        TRAZA += Matriz[i, i]
    return TRAZA


def Vandermonde(N):
    A = np.array([[(i/float(N))**(j-1) for j in range(1, N+1)]
                  for i in range(1, N+1)])
    return A


# Ejemplo de uso
matriz_vandermonde = Vandermonde(4)
resultado_traza = TRAZA(matriz_vandermonde)

print("Matriz Vandermonde(4):")
print(matriz_vandermonde)
print("Traza de la matriz Vandermonde(4):", resultado_traza)


def elevar_matriz(matriz, potencia):
    """
    Eleva una matriz a una potencia dada.

    Parameters:
    matriz (numpy.ndarray): La matriz que se elevará.
    potencia (int): La potencia a la que se elevará la matriz.

    Returns:
    numpy.ndarray: La matriz elevada a la potencia.
    """
    return np.linalg.matrix_power(matriz, potencia)


elevar_matriz(Vandermonde(4), 5)
print(elevar_matriz(Vandermonde(4), 2))
print(matmul(Vandermonde(4), Vandermonde(4)))


def elevars_matriz(matriz, potencia):
    """
    Eleva una matriz a una potencia dada.

    Parameters:
    matriz (numpy.ndarray): La matriz que se elevará.
    potencia (int): La potencia a la que se elevará la matriz.

    Returns:
    numpy.ndarray: La matriz elevada a la potencia.
    """
    if potencia == 0:
        # La matriz elevada a la potencia 0 es la matriz identidad
        return np.eye(matriz.shape[0], dtype=matriz.dtype)

    resultado = np.copy(matriz)
    for _ in range(1, potencia):
        resultado = np.dot(resultado, matriz)

    return resultado


# Ejemplo de uso
print(elevars_matriz(Vandermonde(4), 2))


def Hilbert_matrix(N):
    H = np.array([[1/(i+j-1) for i in range(1, N+1)] for j in range(1, N+1)])
    return H
