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