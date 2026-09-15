# main.py
import time
from machine import ADC, Pin
from extraeremg import ExtractorEMGInferencia
from pca import ReductorPCAInferencia
from Matematicas import kernel_lineal, kernel_rbf

# ==============================================================================
# CONFIGURACIÓN DE MODOS Y PARÁMETROS EXPORTADOS DESDE TU PC
# ==============================================================================
MODO_PRUEBA = True   # True = Procesa una señal estática de 100 datos | False = Lee del ADC físico
TIPO_KERNEL = 'rbf'  # 'rbf' o 'linear' (según con qué entrenaste en tu PC)

# === CONFIGURACIÓN GENERAL ===
GAMMA = 0.2
C_PARAM = 10

# === PARÁMETROS DE PREPROCESAMIENTO (PCA) ===
PCA_MEAN = [0.113896, 0.084678, 0.01142, 157.100006]
PCA_STD = [0.025324, 0.016891, 0.005915, 43.865616]
PCA_AUTOVECTORES = [[0.553548, 0.516263, 0.552017, 0.349763], [-0.199509, -0.437188, 0.054349, 0.875277]]

# === PARÁMETROS DEL MODELO SVM ===
BIAS = -0.332502
LAMBDAS_SV = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.983359, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.45442, 10.0, 10.0, 10.0, 6.026732, 10.0, 10.0, 6.511717, 3.100668, 10.0, 10.0]
ETIQUETAS_SV = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
DATOS_SV = [[0.74387, -0.181446], [0.317006, 0.708413], [0.577743, -0.508686], [-0.524281, 0.744223], [0.619965, -1.073811], [-0.40355, 0.672013], [-0.05602, 1.254626], [0.743118, 0.535973], [-0.479609, 0.227756], [0.738888, -1.023939], [-0.566481, 0.803601], [0.590365, 0.472566], [-0.297531, 0.191961], [0.348035, 0.683042], [-0.195257, 0.206665], [0.280177, 1.765688], [-0.477671, 0.373751], [0.126693, -1.135079], [0.0001, -0.473917], [0.099553, -1.646878], [-0.664883, 0.389598], [0.718726, 0.094586], [-0.281263, 0.971023], [-0.074259, 0.671132]]

# ==============================================================================

def predecir_svm(vector_2d):
    """Calcula la función de decisión evaluando automáticamente el kernel configurado."""
    suma = BIAS
    for lam, etiqueta, x_sv in zip(LAMBDAS_SV, ETIQUETAS_SV, DATOS_SV):
        if TIPO_KERNEL == 'linear':
            k_val = kernel_lineal(x_sv, vector_2d)
        else:
            k_val = kernel_rbf(x_sv, vector_2d, GAMMA)
        suma += lam * etiqueta * k_val
    return -1 if suma >= 0 else 1

def ejecutar_pipeline_emg(senal_cruda, extractor, pca_inf):
    """Procesa una ventana completa de 100 muestras y retorna la clasificación."""
    # 1. Extracción de características (4D: RMS, MAV, VAR, ZCR)
    caracteristicas = extractor.extraer(senal_cruda)
    
    # 2. Reducción PCA (2D: PC1, PC2)
    vector_2d = pca_inf.transformar(caracteristicas)
    
    # 3. Clasificación SVM
    resultado = predecir_svm(vector_2d)
    
    return caracteristicas, vector_2d, resultado

def main():
    print("=== SISTEMA EMBARCADO EMG - RP2040 ===")
    print(f"Modo de operación: {'PRUEBA (Estático)' if MODO_PRUEBA else 'NORMAL (ADC Físico)'}")
    
    # Inicializar módulos con parámetros fijos
    extractor = ExtractorEMGInferencia(v_offset=1.75, umbral_ruido=0.0)
    pca_inf = ReductorPCAInferencia(PCA_MEAN, PCA_STD, PCA_AUTOVECTORES)
    
    if not MODO_PRUEBA:
        # Configurar Pin ADC físico (Ejemplo: Pin 26 / ADC(0))
        adc = ADC(Pin(26))
        print("[+] Lector ADC físico inicializado en Pin 26.")
    
    # --------------------------------------------------------------------------
    # MODO PRUEBA: Inyectamos una señal cruda de prueba (100 muestras)
    # --------------------------------------------------------------------------
    if MODO_PRUEBA:
        print("\n[MODO PRUEBA] Generando señal cruda simulada de 100 muestras...")
        # Simulamos una señal con amplitud alta (ej. simulando Mano Cerrada)
        senal_prueba = [1.743, 1.7341, 1.7292, 1.7295, 1.7322, 1.7364, 1.7448, 1.7562, 1.7613, 1.7617, 1.7661, 1.7707, 1.7706, 1.7726, 1.7701, 1.7597, 1.7528, 1.7531, 1.7519, 1.7471, 1.7423, 1.7435, 1.7461, 1.7497, 1.7588, 1.7668, 1.7748, 1.7816, 1.7831, 1.7827, 1.7809, 1.7763, 1.7715, 1.7631, 1.7555, 1.7515, 1.7433, 1.7398, 1.7449, 1.7471, 1.756, 1.7721, 1.7781, 1.7806, 1.7919, 1.8012, 1.7994, 1.787, 1.7789, 1.7761, 1.7717, 1.7651, 1.7589, 1.7516, 1.7492, 1.7537, 1.7699, 1.7827, 1.7809, 1.7776, 1.7841, 1.7917, 1.796, 1.7992, 1.8038, 1.7972, 1.782, 1.7768, 1.7745, 1.7671, 1.7669, 1.7669, 1.7613, 1.7642, 1.7702, 1.7833, 1.8033, 1.8085, 1.8029, 1.8129, 1.8203, 1.805, 1.7892, 1.7843, 1.7865, 1.7891, 1.7836, 1.7713, 1.7657, 1.7724, 1.7848, 1.7926, 1.7987, 1.807, 1.8154, 1.8169, 1.8115, 1.7969, 1.7913, 1.8039]
        
        print(f"[+] Procesando señal de prueba de {len(senal_prueba)} muestras...")
        caracts, vec2d, pred = ejecutar_pipeline_emg(senal_prueba, extractor, pca_inf)
        
        accion = "MANO CERRADA (1)" if pred == 1 else "MANO ABIERTA (-1)"
        print("\n--------------------------------------------------")
        print("                 RESULTADO DE PRUEBA              ")
        print("--------------------------------------------------")
        print(f"-> Características [RMS, MAV, VAR, ZCR]: {[round(c, 4) for c in caracts]}")
        print(f"-> Proyección PCA [PC1, PC2]:            {[round(p, 4) for p in vec2d]}")
        print(f"-> Predicción Final:                     {accion}")
        print("--------------------------------------------------\n")
        return # Termina la prueba estática

    # --------------------------------------------------------------------------
    # MODO NORMAL: Lectura continua en tiempo real desde el pin ADC
    # --------------------------------------------------------------------------
    print("\n[+] Sistema en vivo. Capturando ventanas de 100 muestras (200 ms)...")
    while True:
        muestras_senal = []
        
        # Tomar 100 muestras con separación de 2ms (Total ventana = 200ms)
        for _ in range(100):
            # Lectura del ADC de la RP2040 convertida a Voltios (asumiendo referencia de 3.3V)
            val_u16 = adc.read_u16()
            voltaje = val_u16 * (3.3 / 65535.0)
            muestras_senal.append(voltaje)
            time.sleep_ms(2)
            
        # Procesar ventana capturada
        caracts, vec2d, pred = ejecutar_pipeline_emg(muestras_senal, extractor, pca_inf)
        accion = "MANO CERRADA (1)" if pred == 1 else "MANO ABIERTA (-1)"
        
        print(f"-> Características: {[round(c,4) for c in caracts]} | PC: {[round(p,4) for p in vec2d]} | Pred: {accion}")

if __name__ == "__main__":
    main()
