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
    def __init__(self, max_iter=100, kernel='rbf', gamma=0.2, C=10.0):
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
        # Asegurar tipo de datos float respetando el valor original (-1 o 1)
        self.y = array.array("f", [float(val) for val in y])
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
        # Retorna 1 para Mano Cerrada y -1 para Mano Abierta
        return [1 if s >= 0 else -1 for s in scores]
    def exportar_parametros(self, ruta="Parametros.py"):
        """Exporta los vectores de soporte y bias en formato Python clásico."""
        sv_indices = [i for i, l in enumerate(self.lambdas) if l > 1e-5]
        
        lambdas_sv = [self.lambdas[i] for i in sv_indices]
        etiquetas_sv = [self.y[i] for i in sv_indices]
        datos_sv = [self.X[i] for i in sv_indices]
        
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(f"BIAS = {self.b}\n")
            f.write(f"LAMBDAS = {lambdas_sv}\n")
            f.write(f"ETIQUETAS = {etiquetas_sv}\n")
            f.write(f"DATOS_SOPORTE = {datos_sv}\n")
            f.write(f"GAMMA = {self.gamma}\n")
            f.write(f"KERNEL = '{self.kernel_type}'\n")
            
        print(f"Parámetros clásicos exportados a '{ruta}'")
    def exportar_parametros_rp2040(self, pca_model, ruta="modelo_exportado.txt", decimales=6):
        """
        Exporta en un único archivo TXT todos los parámetros necesarios para la RP2040:
        - Medias y desviaciones del PCA (Z-score)
        - Autovectores del PCA (Proyección)
        - Lambdas, Etiquetas, Vectores de Soporte (X) y el Bias (b) de la SVM
        - Parámetros de configuración (gamma, C, tipo de kernel)
        """
        sv_indices = [i for i, l in enumerate(self.lambdas) if l > 1e-5]
        
        # Filtrar solo los vectores soporte reales
        lambdas_sv = [round(self.lambdas[i], decimales) for i in sv_indices]
        etiquetas_sv = [round(self.y[i], decimales) for i in sv_indices]
        datos_sv = [[round(elem, decimales) for elem in self.X[i]] for i in sv_indices]
        
        # Parámetros del PCA
        pca_mean = [round(m, decimales) for m in pca_model.mean]
        pca_std = [round(s, decimales) for s in pca_model.std]
        pca_autovectores = [[round(elem, decimales) for elem in vec] for vec in pca_model.autovectores]
        
        Br = round(self.b, decimales)

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("# === CONFIGURACIÓN GENERAL ===\n")
            f.write(f"KERNEL = '{self.kernel_type}'\n")
            f.write(f"GAMMA = {self.gamma}\n")
            f.write(f"C_PARAM = {self.C}\n\n")
            
            f.write("# === PARÁMETROS DE PREPROCESAMIENTO (PCA) ===\n")
            f.write(f"PCA_MEAN = {pca_mean}\n")
            f.write(f"PCA_STD = {pca_std}\n")
            f.write(f"PCA_AUTOVECTORES = {pca_autovectores}\n\n")
            
            f.write("# === PARÁMETROS DEL MODELO SVM ===\n")
            f.write(f"BIAS = {Br}\n")
            f.write(f"LAMBDAS_SV = {lambdas_sv}\n")
            f.write(f"ETIQUETAS_SV = {etiquetas_sv}\n")
            f.write(f"DATOS_SV = {datos_sv}\n")
            
        print(f"¡Parámetros exportados exitosamente a '{ruta}' listos para la RP2040!")

    def exportar_estadisticas_txt(self, y_real, y_pred, ruta="Reporte_Estadisticas.txt"):
        """ Genera un archivo TXT con el resumen de métricas clave del modelo. """
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == -1 and p == -1)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == -1 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == -1)

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