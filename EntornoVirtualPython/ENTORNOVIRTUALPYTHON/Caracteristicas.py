import array

class ExtractorEMG:
    def __init__(self, v_offset=1.75, umbral_ruido=0.01):
        self.v_offset = v_offset
        self.umbral_ruido = umbral_ruido  # Umbral para evitar falsos cruces por ruido estático

    def extraer_4_caracteristicas(self, señal):
        """Extrae [RMS, MAV, VAR, ZCR] centrando la señal respecto a su offset."""
        s_centrada = [v - self.v_offset for v in señal]
        N = len(s_centrada)
        
        if N == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # 1. MAV (Mean Absolute Value)
        mav = sum(abs(x) for x in s_centrada) / N
        
        # 2. RMS (Root Mean Square) - SIN USAR MATH
        rms = (sum(x**2 for x in s_centrada) / N) ** 0.5
        
        # 3. VAR (Variance)
        media = sum(s_centrada) / N
        var = sum((x - media)**2 for x in s_centrada) / (N - 1) if N > 1 else 0.0
        
        # 4. ZCR (Zero Crossings / Cruces por Cero)
        zcr = 0
        for i in range(N - 1):
            # Detecta cambio de signo y asegura que la amplitud supere el ruido mínimo
            if (s_centrada[i] * s_centrada[i+1] < 0) and (abs(s_centrada[i] - s_centrada[i+1]) >= self.umbral_ruido):
                zcr += 1
        
        return [round(rms, 4), round(mav, 4), round(var, 4), float(zcr)]