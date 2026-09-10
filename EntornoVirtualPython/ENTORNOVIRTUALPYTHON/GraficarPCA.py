import numpy as np
import matplotlib.pyplot as plt

def graficar_transicion_4d_a_2d(X_4d, y, X_pca):
    X_4d = np.array(X_4d)
    y = np.array(y)
    X_pca = np.array(X_pca)

    fig = plt.figure(figsize=(14, 6))

    # --- GRAFICA 3D ---
    ax1 = fig.add_subplot(121, projection="3d")
    mask_cerrada = (y == 1)
    mask_abierta = (y == -1)

    # Normalización visual de los diámetros de las esferas
    ssi = X_4d[:, 3]
    ssi_scaled = 20 + 80 * (ssi - ssi.min()) / (ssi.max() - ssi.min() + 1e-8)

    if np.any(mask_cerrada):
        ax1.scatter(X_4d[mask_cerrada, 0], X_4d[mask_cerrada, 1], X_4d[mask_cerrada, 2],
                    c="red", s=ssi_scaled[mask_cerrada], label="Mano Cerrada (1)", alpha=0.6)
    if np.any(mask_abierta):
        ax1.scatter(X_4d[mask_abierta, 0], X_4d[mask_abierta, 1], X_4d[mask_abierta, 2],
                    c="blue", s=ssi_scaled[mask_abierta], label="Mano Abierta (-1)", alpha=0.6)

    ax1.set_title("1. Espacio Real 4D\n(X: RMS, Y: MAV, Z: VAR, Tamaño: SSI)")
    ax1.set_xlabel("RMS")
    ax1.set_ylabel("MAV")
    ax1.set_zlabel("VAR")
    ax1.legend()

    # --- GRAFICA 2D (PCA) ---
    ax2 = fig.add_subplot(122)
    if np.any(mask_cerrada):
        ax2.scatter(X_pca[mask_cerrada, 0], X_pca[mask_cerrada, 1], c="red", label="Mano Cerrada (1)", alpha=0.7)
    if np.any(mask_abierta):
        ax2.scatter(X_pca[mask_abierta, 0], X_pca[mask_abierta, 1], c="blue", label="Mano Abierta (-1)", alpha=0.7)

    ax2.set_title("2. Proyección Reducida PCA (2D)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def graficar_svm_frontera(X_pca, y, svm):
    X_pca = np.array(X_pca)
    y = np.array(y)

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_cerrada = (y == 1)
    mask_abierta = (y == -1)

    # 1. Dibujar datos de entrenamiento
    ax.scatter(X_pca[mask_abierta, 0], X_pca[mask_abierta, 1], c='blue', label='Mano Abierta (-1)', alpha=0.7)
    ax.scatter(X_pca[mask_cerrada, 0], X_pca[mask_cerrada, 1], c='red', label='Mano Cerrada (1)', alpha=0.7)

    # 2. Generar malla
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    puntos_malla = np.c_[XX.ravel(), YY.ravel()].tolist()

    # 3. Predicción evaluando punto por punto
    try:
        predicciones = [svm.predict([pt])[0] for pt in puntos_malla]
        Z = np.array(predicciones).reshape(XX.shape)

        # Contorno de la línea de decisión y regiones de decisión
        ax.contour(XX, YY, Z, levels=[0], colors='black', linewidths=2.5, linestyles='-')
        ax.contourf(XX, YY, Z, levels=[-2, 0, 2], alpha=0.1, colors=['blue', 'red'])
        print("[+] Frontera de decisión calculada y trazada exitosamente.")
    except Exception as e:
        print(f"[-] ERROR crítico al graficar la frontera: {e}")

    ax.set_title("Plano PCA 2D con Frontera de Decisión y Márgenes SVM")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()


def graficar_resultados_test(y_real, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    indices = range(len(y_real))
    
    ax.plot(indices, y_real, 'bo-', label='Real', alpha=0.6)
    ax.plot(indices, y_pred, 'rx--', label='Predicho', alpha=0.8)
    
    ax.set_title("Comparación en Validación / Test")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Clase (1: Cerrada, -1: Abierta)")
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['Abierta (-1)', 'Cerrada (1)'])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()