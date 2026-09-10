"""
from machine import ADC, Pin
import time
adc_pin = 34            # Pin para lectura analógica (ADC1_CH6)
num_muestras = 100    # Número de muestras a tomar
delay_us = 10
voltaje_ref = 3.3
# Tiempo entre muestras en milisegundos
# Inicializar el ADC
adc = ADC(Pin(adc_pin))
adc.atten(ADC.ATTN_11DB)    # Rango de entrada: 0-3.3V
adc.width(ADC.WIDTH_12BIT)  # Resolución de 12 bits (0-4095)
def tomar_muestras():
    # Crear una lista para almacenar las muestras
    muestras = []
    print("Iniciando muestreo...")
    # Tomar las muestras
    for i in range(num_muestras):
        valor = adc.read()      # Leer el valor analógico
        muestras.append(valor)  # Almacenar en la lista
        time.sleep_us(delay_us) # Esperar antes de la siguiente muestra
    print("Muestreo completado.")
    return muestras

print("ESP32 - Muestreo analógico simple")
inicio = time.time()
valores = tomar_muestras()
fin = time.time()

# Imprimir los resultados
print("\nResultados:")
for i, valor in enumerate(valores):
    voltaje = (valor / 4095) * voltaje_ref
    print(f"Muestra {i}: {valor} → {voltaje:.3f} V")
with open("datos.py", "w") as f:
    f.write("muestras = " + str(valores))

print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")
"""
import matplotlib.pyplot as plt
import numpy as np
from DatosEmgCaptadas import KX_train, MX_train

def graficar_senales_emg(num_muestras=3, fs=1000):
    """
    Grafica una comparación entre señales de Mano Cerrada (Activa) 
    y Mano Abierta (Reposo/Inactiva) en el dominio del tiempo.
    """
    # Crear vector de tiempo en segundos (1000 muestras a 1000 Hz = 1.0 s)
    duracion = len(KX_train[0]) / fs
    tiempo = np.linspace(0, duracion, len(KX_train[0]))

    fig, axes = plt.subplots(num_muestras, 2, figsize=(12, 2.5 * num_muestras), sharex=True, sharey=True)

    for i in range(num_muestras):
        # Graficar Mano Cerrada (Columna 1 - Clase 1)
        ax_cerrada = axes[i, 0] if num_muestras > 1 else axes[0]
        ax_cerrada.plot(tiempo, KX_train[i], color='red', alpha=0.8, linewidth=1)
        ax_cerrada.axhline(1.75, color='black', linestyle='--', alpha=0.6, label='Offset (1.75V)')
        ax_cerrada.set_title(f"Mano Cerrada - Muestra {i+1}")
        ax_cerrada.set_ylabel("Voltaje (V)")
        ax_cerrada.grid(True, linestyle="--", alpha=0.5)

        # Graficar Mano Abierta (Columna 2 - Clase -1)
        ax_abierta = axes[i, 1] if num_muestras > 1 else axes[1]
        ax_abierta.plot(tiempo, MX_train[i], color='blue', alpha=0.8, linewidth=1)
        ax_abierta.axhline(1.75, color='black', linestyle='--', alpha=0.6, label='Offset (1.75V)')
        ax_abierta.set_title(f"Mano Abierta - Muestra {i+1}")
        ax_abierta.grid(True, linestyle="--", alpha=0.5)

    # Etiquetas eje X solo en la última fila
    if num_muestras > 1:
        axes[-1, 0].set_xlabel("Tiempo (s)")
        axes[-1, 1].set_xlabel("Tiempo (s)")
    else:
        axes[0].set_xlabel("Tiempo (s)")
        axes[1].set_xlabel("Tiempo (s)")

    plt.suptitle("Comparación de Señales EMG Captadas (Voltaje vs Tiempo)", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Grafica 3 muestras de cada clase
    graficar_senales_emg(num_muestras=3)