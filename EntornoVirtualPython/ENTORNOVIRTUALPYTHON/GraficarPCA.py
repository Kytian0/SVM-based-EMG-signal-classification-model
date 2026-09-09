import matplotlib.pyplot as plt
import numpy as np

def graficar_transicion_4d_a_2d(X_4d, y, X_pca):
    """
    1. Grafica 4D real en un espacio 3D interactivo (X=RMS, Y=MAV, Z=VAR, Tamaño=SSI).
    2. Grafica la proyección reducida en 2D mediante PCA.
    """
    fig = plt.figure(figsize=(16, 7))
    
    # ----------------------------------------------------
    # 1. VISUALIZACIÓN EN ESPACIO 3D + TAMAÑO (4D REAL)
    # ----------------------------------------------------
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Extraer las 4 características
    rms = [p[0] for p in X_4d]
    mav = [p[1] for p in X_4d]
    var = [p[2] for p in X_4d]
    ssi = [p[3] for p in X_4d]
    
    # Normalizar el tamaño de los puntos para la 4ta dimensión (SSI)
    ssi_arr = np.array(ssi)
    ssi_min, ssi_max = ssi_arr.min(), ssi_arr.max()
    rang = (ssi_max - ssi_min) if (ssi_max - ssi_min) != 0 else 1
    tamanos = 30 + ((ssi_arr - ssi_min) / rang) * 170  # Escala de tamaño entre 30 y 200

    # Separar por clases para pintar
    for c, color, label in [(0, 'blue', 'Mano Abierta (0)'), (1, 'red', 'Mano Cerrada (1)')]:
        idx = [i for i in range(len(y)) if y[i] == c]
        ax3d.scatter(
            [rms[i] for i in idx],
            [mav[i] for i in idx],
            [var[i] for i in idx],
            s=[tamanos[i] for i in idx],
            color=color,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5,
            label=label
        )

    ax3d.set_title("1. Espacio Real 4D\n(X: RMS, Y: MAV, Z: VAR, Tamaño esfera: SSI)", fontsize=11, fontweight='bold')
    ax3d.set_xlabel("RMS")
    ax3d.set_ylabel("MAV")
    ax3d.set_zlabel("VAR")
    ax3d.legend(loc='upper left')

    # ----------------------------------------------------
    # 2. PROYECCIÓN REDUCIDA EN PLANO PCA 2D
    # ----------------------------------------------------
    ax2d = fig.add_subplot(1, 2, 2)
    
    x1_c0 = [X_pca[i][0] for i in range(len(y)) if y[i] == 0]
    x2_c0 = [X_pca[i][1] for i in range(len(y)) if y[i] == 0]
    x1_c1 = [X_pca[i][0] for i in range(len(y)) if y[i] == 1]
    x2_c1 = [X_pca[i][1] for i in range(len(y)) if y[i] == 1]

    ax2d.scatter(x1_c0, x2_c0, color='blue', alpha=0.7, label='Mano Abierta (Clase 0)', s=45)
    ax2d.scatter(x1_c1, x2_c1, color='red', alpha=0.7, label='Mano Cerrada (Clase 1)', s=45)
    ax2d.set_title("2. Proyección Reducida PCA (2D)", fontsize=11, fontweight='bold')
    ax2d.set_xlabel("Componente Principal 1 (PC1)")
    ax2d.set_ylabel("Componente Principal 2 (PC2)")
    ax2d.grid(True, linestyle='--', alpha=0.5)
    ax2d.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
def graficar_svm_frontera(X_pca, y, modelo_svm):
    """ Grafica los datos en 2D, la frontera de decisión (función de decisión = 0) 
        y los Vectores de Soporte obtenidos por el modelo. """
    plt.figure(figsize=(8, 6))

    x1 = [p[0] for i, p in enumerate(X_pca) if y[i] == 0]
    y1 = [p[1] for i, p in enumerate(X_pca) if y[i] == 0]
    x2 = [p[0] for i, p in enumerate(X_pca) if y[i] == 1]
    y2 = [p[1] for i, p in enumerate(X_pca) if y[i] == 1]

    plt.scatter(x1, y1, color="blue", label="Mano Abierta", alpha=0.6, s=40)
    plt.scatter(x2, y2, color="red", label="Mano Cerrada", alpha=0.6, s=40)

    # Crear rejilla para evaluar la frontera de decisión
    x_min = min(p[0] for p in X_pca) - 0.5
    x_max = max(p[0] for p in X_pca) + 0.5
    y_min = min(p[1] for p in X_pca) - 0.5
    y_max = max(p[1] for p in X_pca) + 0.5

    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    puntos_grid = [[x, y] for x, y in zip(XX.ravel(), YY.ravel())]
    
    # Calcular valores de la función de decisión
    Z = modelo_svm.decision_function(puntos_grid)
    Z = np.array(Z).reshape(XX.shape)

    # Dibujar la línea hiperplano (Z = 0) y los márgenes (Z = -1, Z = 1)
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

    # Enmarcar los Vectores de Soporte
    sv_indices = [i for i, l in enumerate(modelo_svm.lambdas) if l > 1e-5]
    sv_x = [X_pca[i][0] for i in sv_indices]
    sv_y = [X_pca[i][1] for i in sv_indices]
    plt.scatter(sv_x, sv_y, s=120, facecolors='none', edgecolors='k', linewidths=1.5, label='Vectores de Soporte')

    plt.title("Plano PCA 2D con Frontera de Decisión y Márgenes SVM")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def graficar_resultados_test(y_real, y_pred):
    """ Grafica la comparación punto a punto entre la clase real y la predicción en Test. """
    plt.figure(figsize=(10, 4))
    indices = list(range(len(y_real)))

    plt.plot(indices, y_real, 'bo-', label='Clase Real (Test)', markersize=8, alpha=0.5)
    plt.plot(indices, y_pred, 'rx--', label='Predicción SVM', markersize=6)

    # Marcar desaciertos con una barra vertical
    for i in indices:
        if y_real[i] != y_pred[i]:
            plt.axvline(x=i, color='purple', linestyle=':', alpha=0.8)

    plt.title("Evaluación de Predicciones en el Conjunto de Prueba (Test)")
    plt.xlabel("Índice de Muestra Test")
    plt.ylabel("Clase (0: Abierta | 1: Cerrada)")
    plt.yticks([0, 1], ['Abierta (0)', 'Cerrada (1)'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()