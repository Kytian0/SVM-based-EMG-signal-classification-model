class SVM_Nativa: 
    def __init__(self, lambdas, etiquetas, b, datos, gamma=0.2, kernel="lineal"):
        self.kernel_type = kernel
        self.gamma = gamma
        self.b = round(b, 4)
        
        self.lambdas = lambdas
        self.y = etiquetas
        self.X = datos
        
        # Precálculo hecho a mano: (lambda_i * y_i)
        self.alpha_y = [float(lambdas[i] * etiquetas[i]) for i in range(len(lambdas))]
        self.num_sv = len(self.alpha_y)
        self.euler = 2.718281828459045

    def exp(self, n):
        """Exponencial nativa mediante constante de Euler."""
        return self.euler ** n

    def productopuntoV(self, vector1, vector2):
        """Producto punto entre dos vectores 1D."""
        return sum(float(vector1[i] * vector2[i]) for i in range(len(vector1)))

    def rbf_1d(self, x1, x2):
        """Kernel RBF calculado nativamente entre dos vectores."""
        distancia_cuadrada = sum((float(x1[i]) - float(x2[i])) ** 2 for i in range(len(x1)))
        return self.exp(-self.gamma * distancia_cuadrada)

    def decision_function_muestra(self, x_muestra):
        sumatoria = 0.0

        if self.kernel_type == 'lineal':
            for i in range(self.num_sv):
                k_val = self.productopuntoV(x_muestra, self.X[i])
                sumatoria += self.alpha_y[i] * k_val
        else: # 'rbf'
            for i in range(self.num_sv):
                k_val = self.rbf_1d(x_muestra, self.X[i])
                sumatoria += self.alpha_y[i] * k_val

        return round(sumatoria + self.b, 4)

    def predict(self, x_muestra):
        score = self.decision_function_muestra(x_muestra)
        return 1 if score >= 0.0 else -1