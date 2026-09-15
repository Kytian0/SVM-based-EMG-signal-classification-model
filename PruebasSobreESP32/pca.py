# pcarp.py
from Matematicas import productopuntoV

class ReductorPCAInferencia:
    def __init__(self, mean, std, autovectores):
        self.mean = mean                # Lista de 4 medias
        self.std = std                  # Lista de 4 desviaciones estándar
        self.autovectores = autovectores # Matriz 2x4 (2 componentes, 4 características)

    def transformar(self, caracteristicas_4d):
        """Aplica Z-score y proyecta de 4D a 2D usando los autovectores entrenados."""
        n_features = len(caracteristicas_4d)
        
        # 1. Estandarización Z-score con los valores fijos del entrenamiento
        x_norm = [(caracteristicas_4d[i] - self.mean[i]) / self.std[i] for i in range(n_features)]
        
        # 2. Proyección lineal (Y = X_norm * V^T -> equivalente a producto punto con autovectores)
        proyeccion = []
        for vec in self.autovectores:
            val = productopuntoV(vec, x_norm)
            proyeccion.append(val)
            
        return proyeccion # Devuelve [PC1, PC2]
