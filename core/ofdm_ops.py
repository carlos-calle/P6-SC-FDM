import numpy as np

def modulate_ofdm(input_symbols, N, M):
    """
    Realiza el Mapeo de Subportadoras y la IDFT de tamaño N.
    """
    num_symbols = len(input_symbols)
    
    # 1. Calcular bloques necesarios
    num_blocks = int(np.ceil(num_symbols / M))
    
    # 2. Calcular cuánto relleno falta
    pad_len = num_blocks * M - num_symbols
    
    # 3. Rellenar con ceros (Padding)
    if pad_len > 0:
        # Concatenamos ceros al final para completar el bloque
        input_symbols = np.concatenate((input_symbols, np.zeros(pad_len)))
    
    time_signal = []
    
    for i in range(num_blocks):
        # Tomar un bloque de tamaño M
        block_data = input_symbols[i*M : (i+1)*M]
        
        # Vector de entrada a la IFFT
        ifft_input = np.zeros(N, dtype=complex)
        
        # Insertamos los M datos en las posiciones activas
        # (Saltamos la DC en el índice 0)
        ifft_input[1 : M+1] = block_data 
        
        # IDFT de tamaño N
        block_time = np.fft.ifft(ifft_input) * np.sqrt(N)
        
        time_signal.extend(block_time)
        
    return np.array(time_signal), num_blocks

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



def apply_dft_precoding(symbols, M):
    """
    Aplica la DFT de tamaño M (Precodificación).
    Entrada: Bloques de símbolos en el tiempo.
    Salida: Símbolos en frecuencia (dominio de precodificación).
    """
    num_symbols = len(symbols)
    
    # Calcular bloques de tamaño M
    num_blocks = int(np.ceil(num_symbols / M))
    
    # Padding para completar el último bloque de tamaño M
    pad_len = num_blocks * M - num_symbols
    if pad_len > 0:
        symbols = np.concatenate((symbols, np.zeros(pad_len)))

    precoded_symbols = []
    
    for i in range(num_blocks):
        # Tomamos un bloque de tamaño M
        block_time = symbols[i*M : (i+1)*M]
        
        # DFT de tamaño M
        block_freq = np.fft.fft(block_time) 
        
        precoded_symbols.extend(block_freq)
        
    return np.array(precoded_symbols)

def remove_dft_precoding(symbols, M):
    """IDFT de tamaño M para recuperar símbolos QAM"""
    num_blocks = len(symbols) // M
    decoded = []
    
    for i in range(num_blocks):
        block_freq = symbols[i*M : (i+1)*M]
        
        # IDFT de tamaño M
        block_time = np.fft.ifft(block_freq)
        decoded.extend(block_time)
        
    return np.array(decoded)