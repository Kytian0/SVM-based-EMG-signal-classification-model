
class SVM:
    def __init__(self, lambdas, etiquetas, b, datos, gamma, kernel="lineal"):
        self.kernel = { 'rbf' : lambda x,y: self.rbf(X=x,Y=y),
                        'lineal' : lambda x,y: self.linear(X=x, Y=y)} [kernel]
        self.X=datos
        self.b=round(b, 3)
        self.lambdas= lambdas
        self.y = etiquetas
        self.euler = 2.7182818284590452353602874713527
        self.gamma=gamma
        print("se creo una SVM")
    def exp(self, n):
        return self.euler**n
    def determinar_dimension(self, matriz):
        if not any(isinstance(elemento, list) for elemento in matriz):
            return 1
        longitud_primera_sublista = len(matriz[0])
        if all(isinstance(sublista, list) and len(sublista) == longitud_primera_sublista for sublista in matriz):
            
            return 2
    def productopuntoV(self, vector1=[0], vector2=[0]):
        if len(vector2) == len(vector1):
            return sum(float(vector1[i] * vector2[i]) for i in range(len(vector1)))
        return None
    def transponer_matriz(self, matriz):
        if not matriz:  # Verificar si la matriz está vacía
            return []
        filas = len(matriz)
        columnas = len(matriz[0])
        transpuesta = [[float(matriz[i][j]) for i in range(filas)] for j in range(columnas)]
        return transpuesta
    def ProductoMatrices(self,matriz1=[[0]],matriz2=[[0]]):
        filas_matriz1 = len(matriz1)
        columnas_matriz1 = len(matriz1[0]) 
        filas_matriz2 = len(matriz2)
        columnas_matriz2 = len(matriz2[0]) 
        if columnas_matriz1 != filas_matriz2:
            print("Error: Las dimensiones de las matrices no son compatibles.")
            return None
        resultado = [[0.0 for _ in range(columnas_matriz2)] for _ in range(filas_matriz1)]
        for i in range(filas_matriz1):
            for k in range(columnas_matriz2):
                columnaMatriz2 = [matriz2[j][k] for j in range(filas_matriz2)]
                resultado[i][k] =round( self.productopuntoV(matriz1[i], columnaMatriz2),3)
        return resultado
    def multiplicar_matrices_elemento_a_elemento(self, matriz1, matriz2):
        if len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0]):
            print("Error: Las matrices deben tener las mismas dimensiones.")
            return None
        return [[float(matriz1[i][k] * matriz2[i][k]) for k in range(len(matriz1[0]))] for i in range(len(matriz1))]
    def multiplicar_vectores(self, vector1, vector2):
        if len(vector1) != len(vector2):
            print("Error: Los vectores deben tener la misma longitud.")
            return None
        return [float(vector1[i] * vector2[i]) for i in range(len(vector1))]
    def multiplicacion_elementoV_por_elementoM(self, matriz1, matriz2):
        dim_matriz1 = self.determinar_dimension(matriz1)
        dim_matriz2 = self.determinar_dimension(matriz2)
        if dim_matriz1 == 1 and dim_matriz2 == 1:
            return self.multiplicar_vectores(matriz1, matriz2)       
        elif dim_matriz1 == 1 and dim_matriz2 == 2 and len(matriz1) == len(matriz2[0]):
            return self.multiplicar_vector_por_matriz(matriz1, matriz2)
        elif dim_matriz1 == 2 and dim_matriz2 == 2 and len(matriz1) == len(matriz2) and len(matriz1[0]) == len(matriz2[0]):
            return self.multiplicar_matrices_elemento_a_elemento(matriz1, matriz2)
        else:
            print("Error: Dimensiones incompatibles para la multiplicación.")
            return None
    def multiplicar_vector_por_matriz(self, vector, matriz):
        if len(vector) != len(matriz[0]):
            print("Error: Las dimensiones no son compatibles.")
            return None
        return [[float(vector[k] * matriz[i][k]) for k in range(len(matriz[0]))] for i in range(len(matriz))]
    def sumDatosMatriz(self, datos, columna=None):
        dim = self.determinar_dimension(datos)
        if dim == 1:
            return sum(float(x) for x in datos)
        elif columna == 0:
            return [sum(float(x) for x in fila) for fila in datos]
        elif columna == 1:
            num_columnas = len(datos[0])
            num_filas = len(datos)
            return [sum(float(datos[i][j]) for i in range(num_filas)) for j in range(num_columnas)]
        else:
            print("Error: Especifique una dirección válida para la suma (columna=0 o columna=1)")
            return None     
    def linear(self, X=[], Y=[]):
        resultado = self.ProductoMatrices(matriz1=X, matriz2=self.transponer_matriz(Y))    
        return resultado
    def rbf(self, X=[], Y=[]):
        filas = len(Y)
        fondos = len(X)
        columnas = len(X[0])
        sumas_x = [sum(X[k][j]**2 for j in range(columnas)) for k in range(fondos)]
        sumas_y = [sum(Y[i][j]**2 for j in range(columnas)) for i in range(filas)]
        resultado = [[0.0 for _ in range(filas)] for _ in range(fondos)]
        # Calcular valores del kernel
        for k in range(fondos):
            x_vector = X[k]
            suma_x = sumas_x[k]
            for i in range(filas):
                y_vector = Y[i]
                producto_punto = sum(x_vector[j] * y_vector[j] for j in range(columnas))
                distancia_cuadrada = suma_x + sumas_y[i] - 2 * producto_punto
                resultado[k][i] = round(self.exp(-self.gamma * distancia_cuadrada),3) 
        return resultado
    def float_to_float32(self, f):
        return struct.unpack('f', struct.pack('f', f))[0]
      # Queda como float32 real
    def decision_function(self, X):
        sumatoria = self.sumDatosMatriz(self.multiplicacion_elementoV_por_elementoM( self.lambdas   , self.multiplicacion_elementoV_por_elementoM( self.y   ,  self.kernel(x=X, y=self.X))), columna=0)
        for i in range(len(sumatoria)):
            sumatoria[i]+=self.b
            sumatoria[i]=round(sumatoria[i],4)

        return sumatoria