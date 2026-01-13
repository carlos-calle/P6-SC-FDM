import numpy as np

def modulate_ofdm(symbols, n_fft, nc):
    """
    Empaqueta símbolos en subportadoras y aplica IFFT.
    symbols: Array de símbolos complejos (QPSK/QAM)
    n_fft: Tamaño total de la FFT (ej. 1024)
    nc: Subportadoras activas (ej. 600)
    """
    num_symbols = len(symbols)
    # Número de bloques OFDM necesarios
    num_blocks = int(np.ceil(num_symbols / nc))
    
    # Rellenar con ceros si el último bloque no está lleno
    pad_len = num_blocks * nc - num_symbols
    if pad_len > 0:
        symbols = np.concatenate((symbols, np.zeros(pad_len)))
        
    ofdm_time_signal = []
    
    for i in range(num_blocks):
        # Extraer bloque de símbolos
        block = symbols[i*nc : (i+1)*nc]
        
        # Mapeo a entradas de IFFT
        ifft_input = np.zeros(n_fft, dtype=complex)
        
        # Mapeo simple: Frecuencias 0 a Nc-1 (Parte positiva)
        ifft_input[1:nc+1] = block # Dejamos DC en 0 vacío
        
        # IFFT
        time_sym = np.fft.ifft(ifft_input) * np.sqrt(n_fft) # Normalización de energía
        ofdm_time_signal.extend(time_sym)
        
    return np.array(ofdm_time_signal), num_blocks

def add_cyclic_prefix(signal, num_blocks, n_fft, cp_ratio):
    """Añade el prefijo cíclico a cada bloque OFDM"""
    cp_len = int(n_fft * cp_ratio)
    signal_with_cp = []
    
    # Procesar bloque por bloque
    for i in range(num_blocks):
        block = signal[i*n_fft : (i+1)*n_fft]
        cp = block[-cp_len:] # Copiar el final
        signal_with_cp.extend(np.concatenate((cp, block)))
        
    return np.array(signal_with_cp), cp_len

def remove_cyclic_prefix(rx_signal, n_fft, cp_len):
    """Elimina el CP asumiendo sincronización perfecta"""
    block_len = n_fft + cp_len
    num_blocks = len(rx_signal) // block_len
    rx_no_cp = []
    
    for i in range(num_blocks):
        # Extraer bloque completo con CP
        full_block = rx_signal[i*block_len : (i+1)*block_len]
        # Quedarse solo con la parte útil (quitar CP del inicio)
        useful_part = full_block[cp_len:]
        rx_no_cp.extend(useful_part)
        
    return np.array(rx_no_cp)

def demodulate_ofdm(rx_time_signal, n_fft, nc):
    """Aplica FFT para recuperar símbolos en frecuencia"""
    num_blocks = len(rx_time_signal) // n_fft
    rx_symbols_freq = []
    
    for i in range(num_blocks):
        time_block = rx_time_signal[i*n_fft : (i+1)*n_fft]
        fft_out = np.fft.fft(time_block) / np.sqrt(n_fft) # Normalización inversa
        
        # Extraer las subportadoras de datos (misma lógica que en Tx)
        data_subcarriers = fft_out[1:nc+1]
        rx_symbols_freq.extend(data_subcarriers)
        
    return np.array(rx_symbols_freq)

def equalize_channel(rx_freq_symbols, h_impulse_response, n_fft, nc):
    """
    Ecualizador Zero-Forcing (1-tap).
    Divide la señal recibida por la respuesta del canal en frecuencia.
    
    rx_freq_symbols: Símbolos recibidos tras la FFT
    h_impulse_response: Respuesta al impulso del canal (h) que devolvió el módulo Channel
    """
    # 1. Obtener la respuesta en Frecuencia del canal (H)
    # La FFT de h debe ser del mismo tamaño que la FFT de la señal (N_FFT)
    H_freq = np.fft.fft(h_impulse_response, n_fft)
    
    # 2. Extraer los valores de H correspondientes a las subportadoras de datos
    # (Debemos usar los mismos índices que usamos en modulate_ofdm)
    H_data = H_freq[1:nc+1] 
    
    # 3. Ecualización: Y = X * H + N  ==>  X_est = Y / H
    # Procesamos bloque a bloque porque H se aplica a cada bloque OFDM
    num_blocks = len(rx_freq_symbols) // nc
    equalized_symbols = []
    
    # Evitar división por cero
    threshold = 1e-10
    H_data[np.abs(H_data) < threshold] = threshold
    
    for i in range(num_blocks):
        block_y = rx_freq_symbols[i*nc : (i+1)*nc]
        # División elemento a elemento
        block_x_est = block_y / H_data
        equalized_symbols.extend(block_x_est)
        
    return np.array(equalized_symbols)


# P5/core/ofdm_ops.py (Añadir al final)

def apply_dft_precoding(symbols, nc):
    """
    Aplica DFT (Precodificación) a los símbolos QAM.
    Incluye PADDING automático para evitar errores de dimensión.
    """
    num_symbols = len(symbols)
    # 1. Calcular bloques necesarios
    num_blocks = int(np.ceil(num_symbols / nc))
    # 2. Rellenar con ceros
    pad_len = num_blocks * nc - num_symbols
    if pad_len > 0:
        symbols_padded = np.concatenate((symbols, np.zeros(pad_len)))
    else:
        symbols_padded = symbols

    precoded_symbols = []
    # 3. Procesar bloque a bloque
    for i in range(num_blocks):
        block = symbols_padded[i*nc : (i+1)*nc]
        # FFT para precodificar (Mueve los símbolos QAM de Tiempo a Frecuencia)
        precoded_block = np.fft.fft(block) 
        precoded_symbols.extend(precoded_block)
        
    return np.array(precoded_symbols)

def remove_dft_precoding(equalized_symbols, nc):
    """
    Aplica IDFT para recuperar los símbolos QAM originales.
    """
    num_blocks = len(equalized_symbols) // nc
    decoded_symbols = []
    
    for i in range(num_blocks):
        block = equalized_symbols[i*nc : (i+1)*nc]
        # IFFT para revertir la precodificación
        decoded_block = np.fft.ifft(block)
        decoded_symbols.extend(decoded_block)
        
    return np.array(decoded_symbols)