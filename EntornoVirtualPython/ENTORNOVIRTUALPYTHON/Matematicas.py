import array
import math

EULER = 2.718281828459045

def convert_to_array(datos):  
    if not isinstance(datos[0], (list, array.array)):
        return array.array('f', datos)
    return [array.array('f', row) for row in datos]

def determinar_dimension(datos):
    if not isinstance(datos[0], (list, array.array)):
        return 1
    return 2

def productopuntoV(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

def transponer_matriz(matriz):
    filas = len(matriz)
    columnas = len(matriz[0])
    return [array.array('f', [matriz[i][j] for i in range(filas)]) for j in range(columnas)]

def ProductoMatrices(matriz1, matriz2):
    filas_matriz1 = len(matriz1)
    columnas_matriz2 = len(matriz2[0])
    matriz2transpuesta = transponer_matriz(matriz2)

    resultado = []
    for fila_idx in range(filas_matriz1):
        fila_matriz1 = matriz1[fila_idx]
        fila_resultado = array.array('f', [0.0] * columnas_matriz2)
        for columna_idx in range(columnas_matriz2):
            fila_resultado[columna_idx] = productopuntoV(fila_matriz1, matriz2transpuesta[columna_idx])
        resultado.append(fila_resultado)
    return resultado

def productoMatrizVector(matriz, vector):
    return array.array('f', [productopuntoV(fila, vector) for fila in matriz])

def producto_externo_etiquetas(vector1):
    filas = len(vector1)
    matriz_resultado = []
    for i in range(filas):
        vi = vector1[i]
        fila = array.array('f', [vi * vector1[j] for j in range(filas)])
        matriz_resultado.append(fila)
    return matriz_resultado

def normalizar_datos_matriz(datos):
    matriz_transpuesta = transponer_matriz(datos)
    num_caracteristicas = len(matriz_transpuesta)
    num_muestras = len(matriz_transpuesta[0]) if num_caracteristicas > 0 else 0
    resultado_transpuesta = []
    for caracteristica in matriz_transpuesta:
        media = sum(caracteristica) / num_muestras
        desviacion = (sum([(x_i - media)**2 for x_i in caracteristica]) / num_muestras)**0.5
        desviacion = desviacion if desviacion != 0 else 1e-8
        resultado_transpuesta.append([round((x_i - media) / desviacion, 4) for x_i in caracteristica])
    return tuple(transponer_matriz(tuple(resultado_transpuesta)))

def calcular_covarianza(matriz):
    matriz_transpuesta = transponer_matriz(matriz)
    num_caracteristicas = len(matriz_transpuesta)
    num_muestras = len(matriz_transpuesta[0]) if num_caracteristicas > 0 else 0
    matriz_covarianza = []
    for i in range(num_caracteristicas):
        fila_covarianza = []
        for j in range(num_caracteristicas):
            media_i = sum(matriz_transpuesta[i]) / num_muestras
            media_j = sum(matriz_transpuesta[j]) / num_muestras
            cov = sum([(matriz_transpuesta[i][k] - media_i) * (matriz_transpuesta[j][k] - media_j) for k in range(num_muestras)]) / (num_muestras - 1)
            fila_covarianza.append(round(cov, 4))
        matriz_covarianza.append(tuple(fila_covarianza))
    return tuple(matriz_covarianza)

def multiplicar_vectores(vector1, vector2):
    return array.array('f', [a * b for a, b in zip(vector1, vector2)])

def multiplicar_vector_por_matriz(vector, matriz):
    resultado = []
    columnas = len(matriz[0])
    for fila_matriz in matriz:
        fila = array.array('f', [vector[k] * fila_matriz[k] for k in range(columnas)])
        resultado.append(fila)
    return resultado

def multiplicar_matrices_elemento_a_elemento(matriz1, matriz2):
    resultado = []
    columnas = len(matriz1[0])
    for i in range(len(matriz1)):
        fila = array.array('f', [matriz1[i][k] * matriz2[i][k] for k in range(columnas)])
        resultado.append(fila)
    return resultado

def multiplicacion_elementoV_por_elementoM(matriz1, matriz2):
    dim_matriz1 = determinar_dimension(matriz1)
    dim_matriz2 = determinar_dimension(matriz2)
    
    if dim_matriz1 == 1 and dim_matriz2 == 1:
        return multiplicar_vectores(matriz1, matriz2)       
    elif dim_matriz1 == 1 and dim_matriz2 == 2 and len(matriz1) == len(matriz2[0]):
        return multiplicar_vector_por_matriz(matriz1, matriz2)
    elif dim_matriz1 == 2 and dim_matriz2 == 2 and len(matriz1) == len(matriz2) and len(matriz1[0]) == len(matriz2[0]):
        return multiplicar_matrices_elemento_a_elemento(matriz1, matriz2)
    return None

def extraer_filas_o_datos(matriz, indices):
    dim = determinar_dimension(matriz)
    if dim == 2:
        return [array.array('f', matriz[idx]) for idx in indices]
    elif dim == 1:
        return array.array('f', [matriz[idx] for idx in indices])
    return None

def sumDatosMatriz(datos, columna=None):
    dim = determinar_dimension(datos)
    if dim == 1:
        return sum(datos)
    if columna == 0:
        return array.array('f', [sum(fila) for fila in datos])
    elif columna == 1:
        transpuesta = transponer_matriz(datos)
        return array.array('f', [sum(col) for col in transpuesta])

def resVectores(vector1, vector2):
    return array.array('f', [a - b for a, b in zip(vector1, vector2)])

def productoEscalarVector(vector, escalar):
    return array.array('f', [x * escalar for x in vector])

def clip(funcion, min_val, max_val):
    return array.array('f', [min(max(valor, min_val), max_val) for valor in funcion])

def restrict_to_square(t, v0, u, C=1000):
    eps = 1e-8
    producto_u_t = productoEscalarVector(u, t)
    suma_v0_u_t = array.array('f', [v0[0] + producto_u_t[0], v0[1] + producto_u_t[1]])
    clip_result = clip(suma_v0_u_t, 0, C)
    resta_clip_v0 = array.array('f', [clip_result[0] - v0[0], clip_result[1] - v0[1]])
    
    if abs(u[1]) > eps:
        t = resta_clip_v0[1] / u[1]
    
    producto_u_t = productoEscalarVector(u, t)
    suma_v0_u_t = array.array('f', [v0[0] + producto_u_t[0], v0[1] + producto_u_t[1]])
    clip_result = clip(suma_v0_u_t, 0, C)
    resta_clip_v0 = array.array('f', [clip_result[0] - v0[0], clip_result[1] - v0[1]])
    
    if abs(u[0]) > eps:
        return resta_clip_v0[0] / u[0]
    return t

def extraerIndices(vector, umbral):
    return [i for i in range(len(vector)) if vector[i] > umbral]

def linear(X=[], Y=[]):
    return ProductoMatrices(matriz1=X, matriz2=transponer_matriz(Y))

def rbf(X=[], Y=[], gamma=1):
    fondos = len(X)
    filas = len(Y)
    columnas = len(X[0])
    resultado = []
    
    for k in range(fondos):
        fila_resultado = array.array('f', [0.0] * filas)
        x_vector = X[k]
        for i in range(filas):
            y_vector = Y[i]
            dist_cuadrada = sum((x_vector[j] - y_vector[j]) ** 2 for j in range(columnas))
            fila_resultado[i] = EULER ** (-gamma * dist_cuadrada)
        resultado.append(fila_resultado)
        
    return resultado