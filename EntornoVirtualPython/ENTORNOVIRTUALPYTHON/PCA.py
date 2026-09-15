import array
from Matematicas import transponer_matriz, ProductoMatrices, productoMatrizVector, productopuntoV

class ReductorPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = []
        self.std = []
        self.autovectores = []

    def fit_transform(self, matriz_caracteristicas):
        """Calcula medias, desviaciones, covarianza y autovectores reales dinámicamente."""
        n_samples = len(matriz_caracteristicas)
        n_features = len(matriz_caracteristicas[0])
        
        # 1. Transponer para operar por columnas (características)
        X_t = transponer_matriz(matriz_caracteristicas)
        
        # 2. Calcular Media y Desviación Estándar por característica
        self.mean = array.array('f', [sum(col) / n_samples for col in X_t])
        self.std = array.array('f', [])
        for i in range(n_features):
            var_i = sum((X_t[i][j] - self.mean[i])**2 for j in range(n_samples)) / (n_samples - 1) if n_samples > 1 else 1.0
            std_i = var_i ** 0.5
            self.std.append(std_i if std_i > 1e-8 else 1e-8)
            
        # 3. Estandarización manual (Z-score)
        X_norm = []
        for i in range(n_samples):
            fila = array.array('f', [(matriz_caracteristicas[i][j] - self.mean[j]) / self.std[j] for j in range(n_features)])
            X_norm.append(fila)
            
        # 4. Matriz de Covarianza Exacta: Cov = (1 / (N - 1)) * (X_norm^T * X_norm)
        X_norm_t = transponer_matriz(X_norm)
        cov_work = []
        for i in range(n_features):
            fila_cov = []
            for j in range(n_features):
                cov_ij = sum(X_norm_t[i][k] * X_norm_t[j][k] for k in range(n_samples)) / (n_samples - 1) if n_samples > 1 else 0.0
                fila_cov.append(cov_ij)
            cov_work.append(fila_cov)
            
        # 5. Método de las Potencias con Deflación de Hotelling para extraer autovectores reales
        self.autovectores = []
        for _ in range(self.n_components):
            b_k = array.array('f', [1.0 / (n_features ** 0.5)] * n_features)
            
            for _ in range(100):
                b_next = productoMatrizVector(cov_work, b_k)
                norma = sum(x**2 for x in b_next) ** 0.5
                norma = norma if norma > 1e-8 else 1e-8
                b_k = array.array('f', [x / norma for x in b_next])
                
            self.autovectores.append(b_k)
            
            # Autovalor lambda = b_k^T * Cov * b_k
            cov_b = productoMatrizVector(cov_work, b_k)
            eigenvalue = productopuntoV(b_k, cov_b)
            
            # Deflación
            for i in range(n_features):
                for j in range(n_features):
                    cov_work[i][j] -= eigenvalue * b_k[i] * b_k[j]

        # 6. Proyección final (Y = X_norm * V^T)
        V_T = transponer_matriz(self.autovectores)
        return ProductoMatrices(X_norm, V_T)

    def transform(self, matriz_caracteristicas):
        """Proyecta nuevos datos usando estrictamente la media, desviación y autovectores entrenados."""
        n_samples = len(matriz_caracteristicas)
        n_features = len(matriz_caracteristicas[0])
        
        X_norm = []
        for i in range(n_samples):
            fila = array.array('f', [(matriz_caracteristicas[i][j] - self.mean[j]) / self.std[j] for j in range(n_features)])
            X_norm.append(fila)
            
        V_T = transponer_matriz(self.autovectores)
        return ProductoMatrices(X_norm, V_T)