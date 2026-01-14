import numpy as np
import traceback
# Importamos el NÚCLEO (La física pura)
from core import config, utils, ofdm_ops, channel

class OFDMSimulationManager:
    """
    Recibe parámetros de la GUI, coordina los cálculos matemáticos del Core
    y devuelve resultados limpios para visualizar.
    """
    
    def __init__(self):
        
        pass

    def run_image_transmission(self, image_path, bw_idx, profile_idx, mod_type, snr_db, num_paths, is_scfdm=False):
        """
        Ejecuta la cadena: Imagen -> Scramble -> [Precoding si es SC-FDM] -> OFDM -> Canal -> ...
        """
        try:
            n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
            img_size = 250
            
            # 1. Obtener bits y SCRAMBLING (P5)
            tx_bits_raw, tx_img_matrix = utils.image_to_bits(image_path, img_size)
            tx_bits = utils.apply_scrambling(tx_bits_raw) # <--- Vital
            
            # 2. Mapeo a Símbolos
            tx_symbols = utils.map_bits_to_symbols(tx_bits, mod_type)

            # 3. SC-FDM: Precodificación (P6)
            if is_scfdm:
                tx_symbols = ofdm_ops.apply_dft_precoding(tx_symbols, nc)
            
            # 4. Modulación OFDM
            ofdm_time_signal, num_blocks = ofdm_ops.modulate_ofdm(tx_symbols, n_fft, nc)
            tx_signal_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_time_signal, num_blocks, n_fft, cp_ratio)
            
            # 5. Canal
            rx_signal_cp, h_channel = channel.apply_rayleigh(tx_signal_cp, snr_db, num_taps=num_paths)
            
            # 6. Recepción
            rx_signal_no_cp = ofdm_ops.remove_cyclic_prefix(rx_signal_cp, n_fft, cp_len)
            rx_symbols_distorted = ofdm_ops.demodulate_ofdm(rx_signal_no_cp, n_fft, nc)
            rx_symbols_equalized = ofdm_ops.equalize_channel(rx_symbols_distorted, h_channel, n_fft, nc)
            
            # 7. SC-FDM: Remover Precodificación (P6)
            if is_scfdm:
                rx_symbols_equalized = ofdm_ops.remove_dft_precoding(rx_symbols_equalized, nc)
            
            # 8. Demodulación y DESCRAMBLING (P5)
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_symbols_equalized, mod_type)
            
            # Ajustar longitud y Descramblear
            valid_len = len(tx_bits)
            rx_bits_scrambled = rx_bits_scrambled[:valid_len]
            rx_bits = utils.apply_scrambling(rx_bits_scrambled) # <--- Vital
            
            # 9. Métricas (Comparando con RAW)
            bit_errors = np.sum(tx_bits_raw != rx_bits)
            ber = bit_errors / valid_len
            rx_img_matrix = utils.bits_to_image(rx_bits, img_size)
            
            mode_str = "SC-FDM" if is_scfdm else "OFDM"
            
            return {
                "success": True,
                "tx_image": tx_img_matrix,
                "rx_image": rx_img_matrix,
                "ber": ber,
                "snr": snr_db,
                "info": f"BER: {ber:.5f} | Modo: {mode_str}"
            }

        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def calculate_papr_comparison(self, image_path, bw_idx, profile_idx, mod_type):
        """
        Genera DOS curvas PAPR (OFDM vs SC-FDM) usando los datos de la IMAGEN SCRAMBLEADA.
        """
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
        
        # Usamos la imagen real + Scrambling para que sea realista
        img_size = 1500
        tx_bits_raw, _ = utils.image_to_bits(image_path, img_size)
        tx_bits = utils.apply_scrambling(tx_bits_raw) # <--- Scrambling activado
        
        # 1. Mapear bits scrambleados a símbolos
        syms = utils.map_bits_to_symbols(tx_bits, mod_type)
        
        # --- RAMA 1: OFDM (Estándar) ---
        time_ofdm, num_blocks = ofdm_ops.modulate_ofdm(syms, n_fft, nc)
        
        # --- RAMA 2: SC-FDM (Precodificado) ---
        syms_precoded = ofdm_ops.apply_dft_precoding(syms, nc)
        time_scfdm, _ = ofdm_ops.modulate_ofdm(syms_precoded, n_fft, nc)
        
        # Calcular PAPR bloque a bloque para ambas señales
        # (Reutilizamos la lógica de bloques que ya tenías o calculamos directo)
        papr_ofdm = self._get_papr_values(time_ofdm, n_fft, num_blocks)
        papr_scfdm = self._get_papr_values(time_scfdm, n_fft, num_blocks)
        
        # Generar CCDF
        thresholds = np.linspace(0, 12, 100)
        ccdf_ofdm = self._calculate_ccdf(papr_ofdm, thresholds)
        ccdf_scfdm = self._calculate_ccdf(papr_scfdm, thresholds)
        
        return thresholds, ccdf_ofdm, ccdf_scfdm

    def _get_papr_values(self, time_signal, n_fft, num_blocks):
        """Helper para extraer PAPR de una señal larga"""
        values = []
        for i in range(num_blocks):
            block = time_signal[i*n_fft : (i+1)*n_fft]
            power = np.abs(block)**2
            peak = np.max(power)
            avg = np.mean(power)
            if avg > 0:
                values.append(10 * np.log10(peak / avg))
        return values

    def _calculate_ccdf(self, papr_values, thresholds):
        papr_array = np.array(papr_values)
        ccdf = []
        for x in thresholds:
            prob = np.sum(papr_array > x) / len(papr_array) if len(papr_array) > 0 else 0
            ccdf.append(prob)
        return ccdf

    def calculate_ber_curve(self, image_path, bw_idx, profile_idx, mod_type, num_paths):
        """
        Calcula la curva BER usando los bits de la IMAGEN real.
        """
        snr_range = np.linspace(0, 30, 10) 
        ber_values = []
        
        # Cargar bits de la imagen real ---
        # Usamos el mismo tamaño que en la transmisión normal para coherencia
        img_size = 250 
        # Si la carga falla, el try-except de la UI lo capturará
        tx_bits_raw, _ = utils.image_to_bits(image_path, img_size)
        tx_bits = utils.apply_scrambling(tx_bits_raw)    
        
        # Parámetros físicos
        n_fft, nc, cp_ratio, df = utils.get_ofdm_params(bw_idx, profile_idx)
        
        # Pre-modular la imagen
        tx_syms = utils.map_bits_to_symbols(tx_bits, mod_type)
        
        # Iterar sobre las SNRs
        for snr in snr_range:
            # 1. Modulación
            ofdm_sig, n_blks = ofdm_ops.modulate_ofdm(tx_syms, n_fft, nc)
            tx_cp, cp_len = ofdm_ops.add_cyclic_prefix(ofdm_sig, n_blks, n_fft, cp_ratio)
            
            # 2. Canal (Aplica ruido diferente en cada iteración según la SNR)
            rx_cp, h = channel.apply_rayleigh(tx_cp, snr, num_taps=num_paths)
            
            # 3. Recepción
            rx_no_cp = ofdm_ops.remove_cyclic_prefix(rx_cp, n_fft, cp_len)
            rx_syms = ofdm_ops.demodulate_ofdm(rx_no_cp, n_fft, nc)
            rx_eq = ofdm_ops.equalize_channel(rx_syms, h, n_fft, nc)
            rx_bits_scrambled = utils.demap_symbols_to_bits(rx_eq, mod_type)
            rx_bits_scrambled = rx_bits_scrambled[:len(tx_bits)]
            rx_bits = utils.apply_scrambling(rx_bits_scrambled)

            # 4. Cálculo de BER
            # Recortar bits de relleno si los hay
            valid_len = len(tx_bits_raw)
            if len(rx_bits) > valid_len:
                rx_bits = rx_bits[:valid_len]

            ber = np.sum(tx_bits_raw != rx_bits) / valid_len
            ber_values.append(ber)
            
        return snr_range, ber_values
