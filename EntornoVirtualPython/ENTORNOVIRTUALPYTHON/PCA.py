from Matematicas import normalizar_datos_matriz, calcular_covarianza, transponer_matriz, ProductoMatrices

class ReductorPCA:
    def __init__(self):
        self.autovectores = [
            [0.5, 0.5, 0.5, 0.5],       # PC1: Magnitud/Energía global
            [-0.5, -0.5, 0.707, 0.0]    # PC2: Variabilidad y contraste
        ]

    def fit_transform(self, matriz_caracteristicas):
        """ Normaliza las características e implementa proyección Y = X_norm * V^T """
        X_norm = normalizar_datos_matriz(tuple(matriz_caracteristicas))
        V_T = transponer_matriz(self.autovectores)
        X_pca = ProductoMatrices(X_norm, V_T)
        return X_pca

    def transform(self, matriz_caracteristicas):
        return self.fit_transform(matriz_caracteristicas)