import random
import numpy as np

Fs = 1000
MUESTRAS = 100
NUM_SEÑALES = 100
V_MIN, V_MAX, V_OFFSET = 0.0, 3.3, 1.75

np.random.seed(42)
random.seed(42)


def limitar_voltaje(signal):
    return np.clip(signal, V_MIN, V_MAX)


def generar_mano_abierta():
    t = np.arange(MUESTRAS) / Fs
    offset_real = V_OFFSET + np.random.uniform(-0.05, 0.05)
    f1, f2 = np.random.uniform(50, 120), np.random.uniform(80, 180)
    actividad = 0.025 * np.sin(
        2 * np.pi * f1 * t + np.random.uniform(0, 2 * np.pi)
    ) + 0.015 * np.sin(2 * np.pi * f2 * t + np.random.uniform(0, 2 * np.pi))
    ruido = np.random.normal(0, np.random.uniform(0.015, 0.035), MUESTRAS)
    return limitar_voltaje(offset_real + actividad + ruido)


def generar_actividad_muscular(amplitud, frecuencia_base):
    t = np.arange(MUESTRAS) / Fs
    señal = np.zeros(MUESTRAS)
    for _ in range(np.random.randint(5, 9)):
        freq = frecuencia_base + np.random.uniform(-30, 30)
        fase = np.random.uniform(0, 2 * np.pi)
        amp_u = amplitud * np.random.uniform(0.08, 0.20)
        señal += amp_u * np.sin(2 * np.pi * freq * t + fase)
    return señal + np.random.normal(
        0, amplitud * np.random.uniform(0.08, 0.15), MUESTRAS
    )


def generar_mano_cerrada():
    t = np.arange(MUESTRAS) / Fs
    offset_real = V_OFFSET + np.random.uniform(-0.05, 0.05)
    amplitud, freq_base = np.random.uniform(0.55, 1.15), np.random.uniform(
        70, 140
    )
    posicion, ancho = np.random.choice([0, 1, 2]), np.random.randint(45, 70)

    if posicion == 0:
        centro = np.random.randint(ancho // 2, ancho)
    elif posicion == 1:
        centro = np.random.randint(40, 60)
    else:
        centro = np.random.randint(MUESTRAS - ancho, MUESTRAS - ancho // 2)

    centro = max(ancho // 2, min(MUESTRAS - ancho // 2, centro))
    inicio, final = centro - ancho // 2, centro + ancho // 2

    envelope = np.zeros(MUESTRAS)
    for i in range(inicio, final):
        valor = max(0, np.cos(((abs(i - centro)) / (ancho / 2)) * np.pi / 2))
        envelope[i] = valor * np.random.uniform(0.85, 1.15)

    actividad = generar_actividad_muscular(amplitud, freq_base)
    ruido_reposo = np.random.normal(
        0, np.random.uniform(0.015, 0.035), MUESTRAS
    )
    return limitar_voltaje(offset_real + (actividad * envelope) + ruido_reposo)


# --- GENERACIÓN DE DATOS BASE ---
# Asignamos Etiqueta 0 -> Mano Abierta, Etiqueta 1 -> Mano Cerrada
SeñalesManoAbierta = [
    [round(float(v), 4) for v in generar_mano_abierta()]
    for _ in range(NUM_SEÑALES)
]
SeñalesManoCerrada = [
    [round(float(v), 4) for v in generar_mano_cerrada()]
    for _ in range(NUM_SEÑALES)
]

# --- DIVISIÓN ESTRATIFICADA (80% Train, 20% Test) ---
split_idx = int(NUM_SEÑALES * 0.8)  # 80 muestras

# Datos de Mano Abierta (Clase 0)
abierta_train = SeñalesManoAbierta[:split_idx]
abierta_test = SeñalesManoAbierta[split_idx:]

# Datos de Mano Cerrada (Clase 1)
cerrada_train = SeñalesManoCerrada[:split_idx]
cerrada_test = SeñalesManoCerrada[split_idx:]

# Construcción de arreglos finales
X_train = abierta_train + cerrada_train
y_train = [0] * len(abierta_train) + [1] * len(cerrada_train)

X_test = abierta_test + cerrada_test
y_test = [0] * len(abierta_test) + [1] * len(cerrada_test)

# --- MEZCLADO (SHUFFLE) MANTENIENDO CORRESPONDENCIA CON LAS ETIQUETAS ---
train_pairs = list(zip(X_train, y_train))
random.shuffle(train_pairs)
X_train, y_train = zip(*train_pairs)

test_pairs = list(zip(X_test, y_test))
random.shuffle(test_pairs)
X_test, y_test = zip(*test_pairs)

# Convertir de tuplas de vuelta a listas
X_train, y_train = list(X_train), list(y_train)
X_test, y_test = list(X_test), list(y_test)

# --- IMPRESIÓN PARA COPIAR Y PEGAR EN TUS ARCHIVOS ---
"""print(f"X_train = {X_train}\n")
print(f"y_train = {y_train}\n")
print(f"X_test = {X_test}\n")
print(f"y_test = {y_test}\n")"""