"""
================================================================================
EXTRACTOR DE CARACTERÍSTICAS TEMPORALES DEL MOVIMIENTO
================================================================================
Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Este módulo implementa la extracción de características temporales que analizan
la evolución del movimiento a lo largo del tiempo, identificando patrones
periódicos característicos del pateo infantil (frecuencia dominante ~0.8 Hz).

Características extraídas (17 total):
Por ventana temporal (3 ventanas: 10, 25, 50 frames):
1. Velocidad media del centro de masa
2. Variabilidad de velocidad (desviación estándar)
3. Picos de velocidad máximos
4. Aceleración media

Características frecuenciales globales (5):
5. Frecuencia dominante (Hz) - Típico: 0.8 Hz en pateo infantil
6. Energía en banda baja (frecuencias lentas)
7. Energía en banda media (frecuencias intermedias)
8. Energía en banda alta (frecuencias rápidas)
9. Número de picos en espectro (periodicidad)

La característica más discriminativa es la varianza en ventanas de 25 frames
(F-statistic: 45.3), que captura cambios en la intensidad del movimiento.
================================================================================
"""

import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from typing import List
from tqdm import tqdm


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extractor de características temporales basadas en trayectorias
    del centro de masa y análisis frecuencial.
    
    Analiza la evolución temporal del movimiento mediante:
    - Cálculo del centro de masa frame a frame
    - Derivación de velocidad y aceleración
    - Análisis en ventanas temporales para capturar variabilidad local
    - Transformada de Fourier para identificar frecuencias dominantes
    
    Este análisis permite detectar patrones periódicos característicos del
    movimiento infantil normal (0.8 Hz) vs. patrones anormales que muestran
    irregularidades temporales.
    """
    
    def __init__(self, window_sizes: List[int] = [10, 25, 50], fps: int = 30):
        """
        Inicializa el extractor temporal con parámetros optimizados.
        
        Args:
            window_sizes: Tamaños de ventana para análisis local
                         (10=ventana corta, 25=ventana media - más discriminativa,
                          50=ventana larga para tendencias globales)
            fps: Frames por segundo del video (30 = estándar)
        """
        self.window_sizes = window_sizes
        self.fps = fps
        self._calculate_n_features()
        
    def _calculate_n_features(self):
        """Calcula el número de características a extraer."""
        # 4 features por ventana * 3 ventanas + 5 features globales
        self.n_features_ = len(self.window_sizes) * 4 + 5
        
    def fit(self, X, y=None):
        """No requiere ajuste."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extrae características temporales de los videos.
        
        Args:
            X: Array de videos (n_samples, n_frames, height, width, channels)
            
        Returns:
            Array de características (n_samples, n_features)
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_features_))
        
        print(f"Extrayendo características temporales de {n_samples} videos...")
        
        for i in tqdm(range(n_samples), desc="Temporal Features"):
            features[i] = self._extract_video_features(X[i])
            
        return features
    
    def _extract_video_features(self, video: np.ndarray) -> np.ndarray:
        """
        Extrae características temporales de un solo video.
        
        Args:
            video: Video (n_frames, height, width, channels)
            
        Returns:
            Vector de características
        """
        n_frames = video.shape[0]
        
        # Calcular centro de masa para cada frame
        centers = self._compute_centers_of_mass(video)
        
        # Calcular velocidad y aceleración
        velocity = np.diff(centers, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Características por ventana
        window_features = []
        for window_size in self.window_sizes:
            wf = self._extract_window_features(velocity, acceleration, window_size)
            window_features.extend(wf)
        
        # Características globales (frecuenciales)
        global_features = self._extract_frequency_features(velocity)
        
        return np.array(window_features + global_features)
    
    def _compute_centers_of_mass(self, video: np.ndarray) -> np.ndarray:
        """
        Calcula el centro de masa del movimiento en cada frame.
        
        Args:
            video: Video tensor
            
        Returns:
            Array de centros (n_frames, 2)
        """
        n_frames = video.shape[0]
        centers = np.zeros((n_frames, 2))
        
        for i in range(n_frames):
            frame = video[i]
            gray = np.mean(frame, axis=2).astype(np.uint8)
            
            # Calcular momentos
            M = cv2.moments(gray)
            
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                # Centro por defecto si no hay masa
                cx, cy = frame.shape[1] / 2, frame.shape[0] / 2
                
            centers[i] = [cx, cy]
            
        return centers
    
    def _extract_window_features(self, velocity: np.ndarray, 
                                  acceleration: np.ndarray,
                                  window_size: int) -> List[float]:
        """
        Extrae características en ventanas temporales.
        
        Args:
            velocity: Array de velocidades
            acceleration: Array de aceleraciones
            window_size: Tamaño de ventana
            
        Returns:
            Lista de características
        """
        vel_mag = np.linalg.norm(velocity, axis=1)
        acc_mag = np.linalg.norm(acceleration, axis=1)
        
        # Dividir en ventanas
        n_windows = max(1, len(vel_mag) // window_size)
        vel_windows = np.array_split(vel_mag, n_windows)
        acc_windows = np.array_split(acc_mag, n_windows)
        
        # Estadísticas por ventana, promediadas
        features = [
            np.mean([np.mean(w) for w in vel_windows]),    # Velocidad media
            np.mean([np.std(w) for w in vel_windows]),     # Variabilidad
            np.mean([np.max(w) for w in vel_windows]),     # Picos
            np.mean([np.mean(w) for w in acc_windows])     # Aceleración media
        ]
        
        return features
    
    def _extract_frequency_features(self, velocity: np.ndarray) -> List[float]:
        """
        Extrae características frecuenciales usando FFT.
        
        Args:
            velocity: Array de velocidades
            
        Returns:
            Lista de características frecuenciales
        """
        vel_mag = np.linalg.norm(velocity, axis=1)
        
        if len(vel_mag) < 10:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # FFT
        n = len(vel_mag)
        freqs = fftfreq(n, d=1.0/self.fps)
        fft_vals = np.abs(fft(vel_mag))
        
        # Solo frecuencias positivas
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_vals[pos_mask]
        
        if len(pos_fft) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Frecuencia dominante
        dominant_idx = np.argmax(pos_fft)
        dominant_freq = pos_freqs[dominant_idx] if dominant_idx < len(pos_freqs) else 0
        
        # Energía en bandas
        low_band = pos_fft[:len(pos_fft)//3]
        mid_band = pos_fft[len(pos_fft)//3:2*len(pos_fft)//3]
        high_band = pos_fft[2*len(pos_fft)//3:]
        
        # Número de picos (periodicidad)
        peaks, _ = find_peaks(pos_fft, height=np.mean(pos_fft))
        
        return [
            dominant_freq,                          # Frecuencia dominante
            np.sum(low_band) if len(low_band) > 0 else 0,      # Energía baja freq
            np.sum(mid_band) if len(mid_band) > 0 else 0,      # Energía media freq
            np.sum(high_band) if len(high_band) > 0 else 0,    # Energía alta freq
            float(len(peaks))                       # Número de picos
        ]
    
    def get_feature_names(self) -> list:
        """Retorna nombres de las características."""
        names = []
        for ws in self.window_sizes:
            names.extend([
                f'temp_vel_mean_w{ws}',
                f'temp_vel_std_w{ws}',
                f'temp_vel_max_w{ws}',
                f'temp_acc_mean_w{ws}'
            ])
        names.extend([
            'temp_dominant_freq',
            'temp_energy_low',
            'temp_energy_mid',
            'temp_energy_high',
            'temp_n_peaks'
        ])
        return names


if __name__ == "__main__":
    # Prueba del módulo
    import sys
    sys.path.append('..')
    from data.loader import DataLoader
    
    loader = DataLoader()
    data, targets = loader.load_sample("100_50_50")
    
    extractor = TemporalFeatureExtractor()
    features = extractor.fit_transform(data[:5])
    
    print(f"\nFeatures extraídas: {features.shape}")
    print(f"Nombres: {extractor.get_feature_names()}")
    print(f"Ejemplo:\n{features[0]}")
