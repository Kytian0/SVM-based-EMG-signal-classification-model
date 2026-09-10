import random
from Caracteristicas import ExtractorEMG
from PCA import ReductorPCA
from SVMarray import SVM
from GraficarPCA import (
    graficar_transicion_4d_a_2d,
    graficar_svm_frontera,
    graficar_resultados_test
)
from DatosEmgCaptadas import (
    KX_train, Ky_train, KX_test, Ky_test,
    MX_train, MY_train, MX_test, MY_test
)


def imprimir_resumen_consola(fold_num, X_4d_val, y_val, y_pred, X_pca_val, svm):
    print("\n" + "=" * 80)
    print(f"        CONSOLA DE RESULTADOS Y AUDITORÍA DE DATOS EMG (FOLD {fold_num})        ")
    print("=" * 80)
    
    print(f"\n[+] Total muestras de validación en este fold: {len(y_val)}")
    print(f"[+] Bias (b) del modelo SVM:                   {svm.b:.6f}")
    
    print("\n--------------------------------------------------------------------------------")
    print(f"{'N°':<4} | {'RMS':<8} | {'MAV':<8} | {'VAR':<8} | {'SSI':<8} | {'PC1':<8} | {'PC2':<8} | {'Real':<8} | {'Pred':<8} | {'Estado':<10}")
    print("--------------------------------------------------------------------------------")
    
    aciertos = 0
    for i in range(len(y_val)):
        c_4d = X_4d_val[i]
        c_pca = X_pca_val[i]
        real_lbl = "Cerrada" if y_val[i] == 1 else "Abierta"
        pred_lbl = "Cerrada" if y_pred[i] == 1 else "Abierta"
        
        if y_val[i] == y_pred[i]:
            estado = "OK"
            aciertos += 1
        else:
            estado = "ERROR"

        print(f"{i+1:<4} | {c_4d[0]:<8.4f} | {c_4d[1]:<8.4f} | {c_4d[2]:<8.4f} | {c_4d[3]:<8.4f} | {c_pca[0]:<8.4f} | {c_pca[1]:<8.4f} | {real_lbl:<8} | {pred_lbl:<8} | {estado:<10}")

    exactitud = (aciertos / len(y_val)) * 100
    print("--------------------------------------------------------------------------------")
    print(f">> RENDIMIENTO FOLD {fold_num}: {exactitud:.2f}% de exactitud ({aciertos}/{len(y_val)} aciertos)")
    print("=" * 80 + "\n")
    return exactitud


def dividir_en_folds(X, y, k=5, seed=42):
    if len(X) != len(y):
        raise ValueError(f"Error de dimensión: X tiene {len(X)} elementos, pero y tiene {len(y)} elementos.")

    indices = list(range(len(X)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    tamano_fold = len(X) // k
    folds = []
    
    for i in range(k):
        inicio = i * tamano_fold
        fin = len(X) if i == k - 1 else (i + 1) * tamano_fold
        
        idx_val = indices[inicio:fin]
        idx_train = indices[:inicio] + indices[fin:]
        
        X_tr = [X[j] for j in idx_train]
        y_tr = [y[j] for j in idx_train]
        X_val = [X[j] for j in idx_val]
        y_val = [y[j] for j in idx_val]
        
        folds.append((X_tr, y_tr, X_val, y_val))
        
    return folds


def main():
    print("=== INICIANDO PIPELINE EMG CON VALIDACIÓN CRUZADA (5-FOLD) ===")

    # 1. Unificación de los datos reales
    X_completo = KX_train + KX_test + MX_train + MX_test
    y_completo = Ky_train + Ky_test + MY_train + MY_test

    print(f"\n[+] Datos cargados correctamente:")
    print(f"    - Mano Cerrada (K): {len(KX_train) + len(KX_test)} muestras")
    print(f"    - Mano Abierta (M): {len(MX_train) + len(MX_test)} muestras")
    print(f"    - Total global:     {len(X_completo)} señales | {len(y_completo)} etiquetas")

    # 2. Generación de Folds con semilla fija
    folds = dividir_en_folds(X_completo, y_completo, k=5, seed=42)
    extractor = ExtractorEMG(v_offset=1.75)
    
    exactitudes = []
    mejores_modelos = []

    # 3. Ciclo de validación cruzada 5-Fold
    for fold_num, (X_tr, y_tr, X_val, y_val) in enumerate(folds, start=1):
        print(f"\n==================== PROCESANDO FOLD {fold_num} ====================")

        # Extracción de características temporales
        X_tr_4d = [extractor.extraer_4_caracteristicas(sig) for sig in X_tr]
        X_val_4d = [extractor.extraer_4_caracteristicas(sig) for sig in X_val]

        # Reducción de dimensionalidad (4D -> 2D)
        pca = ReductorPCA()
        X_tr_pca = pca.fit_transform(X_tr_4d)
        X_val_pca = pca.transform(X_val_4d)

        # Entrenar SVM con suficiente número de iteraciones
        svm = SVM(max_iter=10, kernel='lineal', C=10)
        svm.fit(X_tr_pca, y_tr)

        # Predicción
        y_pred = svm.predict(X_val_pca)

        # Presentación de métricas
        acc_fold = imprimir_resumen_consola(fold_num, X_val_4d, y_val, y_pred, X_val_pca, svm)
        exactitudes.append(acc_fold)

        mejores_modelos.append({
            'acc': acc_fold,
            'svm': svm,
            'X_tr_4d': X_tr_4d,
            'y_tr': y_tr,
            'X_tr_pca': X_tr_pca,
            'X_val_pca': X_val_pca,
            'y_val': y_val,
            'y_pred': y_pred
        })

    # 4. Resumen general
    acc_promedio = sum(exactitudes) / len(exactitudes)
    print("\n" + "#" * 80)
    print(f"   RESUMEN FINAL K-FOLD: EXACTITUD PROMEDIO GLOBAL = {acc_promedio:.2f}%   ")
    print("#" * 80)
    for i, acc in enumerate(exactitudes, start=1):
        print(f" -> Fold {i}: {acc:.2f}%")
    print("#" * 80 + "\n")

    # 5. Selección y exportación del mejor modelo
    mejores_modelos.sort(key=lambda item: item['acc'], reverse=True)
    best = mejores_modelos[0]

    print(f"[+] Seleccionado el mejor modelo (Fold con {best['acc']:.2f}% de exactitud) para graficación y exportación.")

    # Generación de gráficas
    graficar_transicion_4d_a_2d(best['X_tr_4d'], best['y_tr'], best['X_tr_pca'])
    graficar_svm_frontera(best['X_tr_pca'], best['y_tr'], best['svm'])
    graficar_resultados_test(best['y_val'], best['y_pred'])

    # Exportar parámetros
    best['svm'].exportar_parametros("Parametros.py")
    best['svm'].exportar_estadisticas_txt(best['y_val'], best['y_pred'], "Reporte_Estadisticas.txt")
    print("\n[+] Archivos 'Parametros.py' y 'Reporte_Estadisticas.txt' exportados exitosamente.")


if __name__ == "__main__":
    main()