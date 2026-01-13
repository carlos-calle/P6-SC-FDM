import numpy as np
import cv2
from .config import LTE_BANDWIDTHS, LTE_PROFILES, MOD_CONSTELLATIONS

def get_ofdm_params(bw_idx, profile_idx):
    """Devuelve parámetros físicos calculados"""
    bw_hz, nc = LTE_BANDWIDTHS[bw_idx]
    df, cp_ratio = LTE_PROFILES[profile_idx]
    # N (Tamaño FFT) es la potencia de 2 siguiente a Nc
    n_fft = int(2**(np.ceil(np.log2(nc))))
    return n_fft, nc, cp_ratio, df

def image_to_bits(image_path, size):
    """Convierte imagen a flujo de bits"""
    img = cv2.imread(image_path, 0) # Leer en escala de grises
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    img = cv2.resize(img, (size, size))
    # Convertir a bits
    bits = np.unpackbits(img)
    return bits, img

def bits_to_image(bits, size):
    """Reconstruye imagen desde bits"""
    # Asegurar que la longitud de bits sea correcta para la imagen
    expected_len = size * size * 8
    bits = bits[:expected_len] 
    img = np.packbits(bits)
    img = img.reshape((size, size))
    return img

def get_constellation_map(mod_type):
    """Devuelve el diccionario de mapeo y bits por símbolo"""
    if mod_type == 1: # QPSK
        return MOD_CONSTELLATIONS[1], 2
    elif mod_type == 2: # 16-QAM
        return MOD_CONSTELLATIONS[2], 4
    elif mod_type == 3: # 64-QAM
        return MOD_CONSTELLATIONS[3], 6
    else:
        raise ValueError("Modulación no soportada")

def map_bits_to_symbols(bits, mod_type):
    """Modulador Digital: Bits -> Símbolos Complejos"""
    '''
    Convierte bits a símbolos y NORMALIZA la energía a 1.
    '''
    constellation, n_bits = get_constellation_map(mod_type)
    
    # Rellenar con ceros si falta (padding)
    remainder = len(bits) % n_bits
    if remainder != 0:
        bits = np.concatenate((bits, np.zeros(n_bits - remainder, dtype=int)))
    
    norm_factors = {
        1: 1.0 / np.sqrt(2),
        2: 1.0 / np.sqrt(10),
        3: 1.0 / np.sqrt(42)
    }
    scale = norm_factors[mod_type]
    symbols = []
    # Procesar en chunks
    for i in range(0, len(bits), n_bits):
        chunk = tuple(bits[i:i+n_bits])
        
        sym_val = constellation.get(chunk, 0+0j) * scale
        symbols.append(sym_val)
        
    return np.array(symbols)

def demap_symbols_to_bits(symbols, mod_type):
    """Demodulador Digital (ML): Símbolos Complejos -> Bits"""
    constellation, _ = get_constellation_map(mod_type)
    bits = []

    norm_factors = {
        1: 1.0 / np.sqrt(2),
        2: 1.0 / np.sqrt(10),
        3: 1.0 / np.sqrt(42)
    }
    scale = norm_factors[mod_type]
    
    # Pre-calcular puntos de constelación para búsqueda rápida
    points = np.array(list(constellation.values())) * scale
    bit_maps = list(constellation.keys())
    
    for sym in symbols:
        # Distancia Euclideana |x - y|^2
        distances = np.abs(sym - points)
        min_idx = np.argmin(distances)
        bits.extend(bit_maps[min_idx])
        
    return np.array(bits)


def apply_scrambling(bits, seed=2024):
    """
    Aplica (o revierte) un scrambling aditivo usando una secuencia pseudo-aleatoria.
    Al ser una operación XOR, esta misma función sirve para Scramble y Descramble
    siempre que se use la misma semilla (seed).
    """
    # Guardamos el estado actual del generador random de numpy para no afectar otras partes
    state = np.random.get_state()
    
    try:
        np.random.seed(seed)
        # Generamos una máscara aleatoria del mismo tamaño que los bits
        scrambling_sequence = np.random.randint(0, 2, len(bits))
        # Operación XOR (suma módulo 2)
        scrambled_bits = np.bitwise_xor(bits, scrambling_sequence)
        return scrambled_bits
    finally:
        # Restauramos el estado del generador
        np.random.set_state(state)