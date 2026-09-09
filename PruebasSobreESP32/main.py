from EntornoVirtualPython.ENTORNOVIRTUALPYTHON.Parametros import Lambdas,Etiquetas,ParametroB,Datos
from SVMarray import SVM
from machine import Pin
import time
# Configuramos el pin 25 como salida (LED integrado)
led = Pin(25, Pin.OUT)

inicio = time.time()
led.toggle() 
prueba = [Datos[17], Datos[18],Datos[19], Datos[20]]
svmESP = SVM(Lambdas, Etiquetas,ParametroB,Datos, gamma=1,kernel="rbf")
print(svmESP.decision_function(prueba))
print(len(Datos[0])*len(Datos)* 4)
fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")