# matematicarp.py

def productopuntoV(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def productoMatrizVector(matriz, vector):
    resultado = []
    for fila in matriz:
        resultado.append(productopuntoV(fila, vector))
    return resultado

def kernel_lineal(x1, x2):
    """Calcula el kernel lineal: producto punto directo."""
    return productopuntoV(x1, x2)

def kernel_rbf(x1, x2, gamma):
    """Calcula el kernel RBF sin usar la librería math."""
    dist_sq = sum((a - b) ** 2 for a, b in zip(x1, x2))
    x = -gamma * dist_sq
    # Aproximación de e^x por serie de Taylor
    return 1.0 + x + (x**2)/2.0 + (x**3)/6.0 + (x**4)/24.0 if x >= -10 else 0.0
