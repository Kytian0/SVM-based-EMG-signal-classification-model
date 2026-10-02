# extraccionemgrp.py

class ExtractorEMGInferencia:
    def __init__(self, v_offset=1.75, umbral_ruido=0.01):
        self.v_offset = v_offset
        self.umbral_ruido = umbral_ruido

    def extraer_4_caracteristicas(self, señal):
        """Extrae [Valor Absoluto, MAV, RMS, ZCR] centrando la señal respecto a su offset."""
        s_centrada = [v - self.v_offset for v in señal]
        N = len(s_centrada)
        
        if N == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # 1. Valor Absoluto (Suma de valores absolutos / SAV)
        valor_absoluto = sum(abs(x) for x in s_centrada)
        
        # 2. Valor Absoluto Medio (MAV - Mean Absolute Value)
        mav = valor_absoluto / N
        
        # 3. RMS (Root Mean Square) - Sin usar la librería math
        rms = (sum(x**2 for x in s_centrada) / N) ** 0.5
        
        # 4. ZCR (Zero Crossings / Cruces por Cero)
        zcr = 0
        for i in range(N - 1):
            # Detecta cambio de signo y asegura que la amplitud supere el umbral de ruido
            if (s_centrada[i] * s_centrada[i+1] < 0) and (abs(s_centrada[i] - s_centrada[i+1]) >= self.umbral_ruido):
                zcr += 1
        
        return [round(valor_absoluto, 4), round(mav, 4), round(rms, 4), float(zcr)]
