import os
import numpy as np
from pathlib import Path


def generar_senal_emg_realista(
    duracion_s=2.0,
    fs=1000,
    offset=1.75,
    v_peak=1.2,
    ruido_60hz=True,
    deriva_linea_base=True,
    picos_artefacto=True,
    seed=None
):
    """
    Genera una señal EMG realista filtrada con offset de 1.75V.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duracion_s, int(fs * duracion_s), endpoint=False)
    n_muestras = len(t)

    # Envolvente de activación muscular con protección antidesbordamiento
    centro = duracion_s / 2.0
    ancho = duracion_s / 4.0
    arg1 = np.clip(-12 * (t - (centro - ancho/2)), -500, 500)
    arg2 = np.clip(-12 * (t - (centro + ancho/2)), -500, 500)
    
    envolvente = 1.0 / (1.0 + np.exp(arg1)) * (1.0 - 1.0 / (1.0 + np.exp(arg2)))

    # Ruido estocástico base
    emg_raw = np.random.normal(0, 1, n_muestras)

    # Filtro de respuesta para densidad espectral muscular (20Hz - 450Hz)
    kernel = np.array([-0.05, -0.1, 0.4, 0.8, 0.4, -0.1, -0.05])
    emg_filtrado = np.convolve(emg_raw, kernel, mode='same')
    
    emg_activo = (emg_filtrado / np.max(np.abs(emg_filtrado))) * envolvente * v_peak

    # Artefactos analógicos no ideales
    artefactos = np.zeros(n_muestras)

    if ruido_60hz:
        amplitud_red = np.random.uniform(0.02, 0.08)
        fase = np.random.uniform(0, 2 * np.pi)
        artefactos += amplitud_red * np.sin(2 * np.pi * 60 * t + fase)

    if deriva_linea_base:
        f_deriva = np.random.uniform(0.2, 0.8)
        amp_deriva = np.random.uniform(0.03, 0.12)
        artefactos += amp_deriva * np.sin(2 * np.pi * f_deriva * t)

    if picos_artefacto and np.random.rand() > 0.4:
        idx_pico = np.random.randint(0, n_muestras)
        duracion_pico = int(fs * 0.02)
        fin = min(idx_pico + duracion_pico, n_muestras)
        artefactos[idx_pico:fin] += np.random.uniform(-0.3, 0.3)

    # Superposición final + Offset estricto de 1.75V
    senal_salida = offset + emg_activo + artefactos
    senal_salida = np.clip(senal_salida, 0.0, 3.5)

    return np.round(senal_salida, 4).tolist()


def generar_dataset_emg(num_muestras_por_clase=50):
    KX_totales, MX_totales = [], []

    for _ in range(num_muestras_por_clase):
        v_p = np.random.uniform(0.9, 1.25)
        KX_totales.append(generar_senal_emg_realista(duracion_s=1.0, fs=1000, offset=1.75, v_peak=v_p))

    for _ in range(num_muestras_por_clase):
        v_p = np.random.uniform(0.2, 0.92)
        MX_totales.append(generar_senal_emg_realista(duracion_s=1.0, fs=1000, offset=1.75, v_peak=v_p))

    corte = int(num_muestras_por_clase * 0.8)

    KX_train, KX_test = KX_totales[:corte], KX_totales[corte:]
    MX_train, MX_test = MX_totales[:corte], MX_totales[corte:]

    Ky_train = [1] * len(KX_train)
    Ky_test = [1] * len(KX_test)
    MY_train = [-1] * len(MX_train)
    MY_test = [-1] * len(MX_test)

    return KX_train, Ky_train, KX_test, Ky_test, MX_train, MY_train, MX_test, MY_test


def exportar_a_fichero_python(nombre_archivo="DatosEmgCaptadas.py"):
    ruta_directorio = Path(__file__).resolve().parent
    ruta_completa = ruta_directorio / nombre_archivo

    KX_tr, Ky_tr, KX_te, Ky_te, MX_tr, MY_tr, MX_te, MY_te = generar_dataset_emg(num_muestras_por_clase=50)

    with open(ruta_completa, "w", encoding="utf-8") as f:
        f.write("# Archivo DatosEmgCaptadas.py generado automáticamente\n\n")
        f.write(f"KX_train = {KX_tr}\n\n")
        f.write(f"Ky_train = {Ky_tr}\n\n")
        f.write(f"KX_test = {KX_te}\n\n")
        f.write(f"Ky_test = {Ky_te}\n\n")
        f.write(f"MX_train = {MX_tr}\n\n")
        f.write(f"MY_train = {MY_tr}\n\n")
        f.write(f"MX_test = {MX_te}\n\n")
        f.write(f"MY_test = {MY_te}\n")

    print(f"[+] Archivo guardado correctamente en:\n    {ruta_completa}")


if __name__ == "__main__":
    exportar_a_fichero_python("DatosEmgCaptadas.py")