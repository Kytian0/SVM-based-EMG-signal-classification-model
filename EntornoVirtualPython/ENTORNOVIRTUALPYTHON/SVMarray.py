import array
import random
import time
from Matematicas import (
    convert_to_array, rbf, linear, producto_externo_etiquetas,
    multiplicacion_elementoV_por_elementoM, extraer_filas_o_datos,
    sumDatosMatriz, productoMatrizVector, productopuntoV,
    restrict_to_square, productoEscalarVector, extraerIndices
)

class SVM:
    def __init__(self, max_iter=10, kernel='rbf', gamma=0.2, C=10.0):
        self.max_iter = max_iter
        self.C = C
        self.gamma = gamma
        self.kernel_type = kernel
        
        if kernel == 'rbf':
            self.kernel = lambda x, y: rbf(X=x, Y=y, gamma=self.gamma)
        else:
            self.kernel = lambda x, y: linear(X=x, Y=y)

    def fit(self, X, y):
        start_time = time.time()
        self.X = convert_to_array(X)
        self.y = array.array("f", [(val * 2) - 1 for val in y])
        self.lambdas = array.array("f", [0.0] * len(y))

        kernel_matrix = self.kernel(self.X, self.X)
        prod_ext = producto_externo_etiquetas(self.y)
        self.K = multiplicacion_elementoV_por_elementoM(matriz1=kernel_matrix, matriz2=prod_ext)

        num_samples = len(self.lambdas)

        for iteracion in range(self.max_iter):
            for idxM in range(num_samples):
                idxL = random.randint(0, num_samples - 1)
                while idxL == idxM:
                    idxL = random.randint(0, num_samples - 1)

                Q = [
                    array.array("f", [self.K[idxM][idxM], self.K[idxM][idxL]]),
                    array.array("f", [self.K[idxL][idxM], self.K[idxL][idxL]])
                ]

                v0 = array.array("f", [self.lambdas[idxM], self.lambdas[idxL]])
                filas_K = extraer_filas_o_datos(self.K, [idxM, idxL])

                temp = multiplicacion_elementoV_por_elementoM(self.lambdas, filas_K)
                k0 = sumDatosMatriz(temp, columna=0)

                for i in range(len(k0)):
                    k0[i] = 1.0 - k0[i]

                u = array.array("f", [-self.y[idxL], self.y[idxM]])
                Qu = productoMatrizVector(Q, u)

                denominador = productopuntoV(Qu, u) + 1e-15
                t_max = productopuntoV(k0, u) / denominador

                t_restricted = restrict_to_square(t_max, v0, u, C=self.C)
                delta = productoEscalarVector(u, t_restricted)
                nuevos_valores = sumDatosMatriz([v0, delta], columna=1)
                
                self.lambdas[idxM] = nuevos_valores[0]
                self.lambdas[idxL] = nuevos_valores[1]

        self.tiempo_entrenamiento = time.time() - start_time
        idx_sv = extraerIndices(self.lambdas, 1e-5)
        
        if not idx_sv:
            self.b = 0.0
        else:
            extraerfilas = extraer_filas_o_datos(self.K, idx_sv)
            multiplicacion = multiplicacion_elementoV_por_elementoM(self.lambdas, extraerfilas)
            sumatoria = sumDatosMatriz(multiplicacion, columna=0)

            for i in range(len(sumatoria)):
                sumatoria[i] = 1.0 - sumatoria[i]

            etiquetas_idx = extraer_filas_o_datos(self.y, idx_sv)
            muletiquetas = multiplicacion_elementoV_por_elementoM(sumatoria, etiquetas_idx)
            self.b = sum(muletiquetas) / len(idx_sv)

    def decision_function(self, X):
        X_array = convert_to_array(X)
        kernel_result = self.kernel(x=X_array, y=self.X)
        temp1 = multiplicacion_elementoV_por_elementoM(self.y, kernel_result)
        temp2 = multiplicacion_elementoV_por_elementoM(self.lambdas, temp1)
        sumatoria = sumDatosMatriz(temp2, columna=0)
        return array.array('f', [val + self.b for val in sumatoria])

    def predict(self, X):
        scores = self.decision_function(X)
        return [1 if s >= 0 else 0 for s in scores]

    def exportar_parametros(self, ruta="Parametros.py", decimales=6):
        sv_indices = [i for i, l in enumerate(self.lambdas) if l > 1e-5]
        lambdas_sv = [round(self.lambdas[i], decimales) for i in sv_indices]
        etiquetas_sv = [round(self.y[i], decimales) for i in sv_indices]
        datos_sv = [[round(elem, decimales) for elem in self.X[i]] for i in sv_indices]
        Br = round(self.b, decimales)

        with open(ruta, "w") as f:
            f.write(f"Lambdas = {lambdas_sv}\n")
            f.write(f"Etiquetas = {etiquetas_sv}\n")
            f.write(f"Datos = {datos_sv}\n")
            f.write(f"ParametroB = {Br}\n")

    def exportar_estadisticas_txt(self, y_real, y_pred, ruta="Reporte_Estadisticas.txt"):
        """ Genera un archivo TXT con el resumen de métricas clave del modelo. """
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)

        total = len(y_real)
        accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        sensibilidad = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        especificidad = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

        sv_indices = [i for i, l in enumerate(self.lambdas) if l > 1e-5]

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write("    REPORTE DE ESTADÍSTICAS DEL MODELO SVM (EMG)    \n")
            f.write("====================================================\n\n")
            f.write("--- CONFIGURACIÓN DEL MODELO ---\n")
            f.write(f"Tipo de Kernel:            {self.kernel_type.upper()}\n")
            f.write(f"Parámetro C (Caja):         {self.C}\n")
            f.write(f"Parámetro Gamma:           {self.gamma}\n")
            f.write(f"Iteraciones Máximas:       {self.max_iter}\n")
            f.write(f"Tiempo de Entrenamiento:   {self.tiempo_entrenamiento:.4f} segundos\n\n")

            f.write("--- SOPORTE VECTORIAL ---\n")
            f.write(f"Total Muestras Entrenamiento: {len(self.y)}\n")
            f.write(f"Cantidad de Vectores Soporte: {len(sv_indices)}\n")
            f.write(f"Valor Bias (b):               {self.b:.6f}\n\n")

            f.write("--- RENDIMIENTO EN CONJUNTO DE PRUEBA (TEST) ---\n")
            f.write(f"Total Muestras Evaluación:  {total}\n")
            f.write(f"Verdaderos Positivos (TP): {tp}\n")
            f.write(f"Verdaderos Negativos (TN): {tn}\n")
            f.write(f"Falsos Positivos (FP):     {fp}\n")
            f.write(f"Falsos Negativos (FN):     {fn}\n\n")

            f.write("--- MÉTRICAS GLOBALES ---\n")
            f.write(f"Exactitud (Accuracy):      {accuracy:.2f}%\n")
            f.write(f"Precisión (Precision):     {precision:.2f}%\n")
            f.write(f"Sensibilidad (Recall):     {sensibilidad:.2f}%\n")
            f.write(f"Especificidad:             {especificidad:.2f}%\n")
            f.write("====================================================\n")