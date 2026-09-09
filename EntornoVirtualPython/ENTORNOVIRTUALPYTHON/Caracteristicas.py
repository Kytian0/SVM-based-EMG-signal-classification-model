import math

class ExtractorEMG:
    def __init__(self, v_offset=1.75):
        self.v_offset = v_offset

    def extraer_4_caracteristicas(self, señal):
        """Extrae [RMS, MAV, VAR, SSI] centrando la señal respecto a su offset."""
        s_centrada = [v - self.v_offset for v in señal]
        N = len(s_centrada)
        
        mav = sum(abs(x) for x in s_centrada) / N
        rms = math.sqrt(sum(x**2 for x in s_centrada) / N)
        media = sum(s_centrada) / N
        var = sum((x - media)**2 for x in s_centrada) / (N - 1) if N > 1 else 0.0
        ssi = sum(x**2 for x in s_centrada)
        
        return [round(rms, 4), round(mav, 4), round(var, 4), round(ssi, 4)]