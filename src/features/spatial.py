"""
================================================================================
EXTRACTOR DE CARACTERÍSTICAS ESPACIALES DEL MOVIMIENTO
================================================================================
Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Este módulo implementa la extracción de características espaciales que evalúan
la distribución del movimiento en el espacio, identificando patrones de
simetría y localización característicos del movimiento infantil.

Características extraídas (8 total con n_quadrants=4):
1-4. Actividad por cuadrante (superior izq, superior der, inferior izq, inferior der)
5. Simetría bilateral (horizontal) - Típico: 0.73 en movimientos coordinados
6. Área de movimiento significativo - Proporción del frame con movimiento activo
7. Dispersión espacial (entropía normalizada) - Complejidad del patrón espacial
8. Centro de movimiento X normalizado - Localización horizontal del movimiento

La asimetría en cuadrantes superiores vs inferiores (ratio 1.43) sugiere patrones
de pateo con predominancia de extremidades inferiores, característico de
movimientos infantiles normales.

Simetría bilateral cercana a 1.0 indica movimientos coordinados, mientras que
valores bajos pueden señalar asimetrías motoras asociadas a patología.
================================================================================
"""

import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class SpatialFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extractor de características espaciales basadas en la distribución
    del movimiento en diferentes regiones del frame.
    
    Analiza la distribución espacial del movimiento mediante:
    - Mapa de actividad acumulada a lo largo del video
    - División en cuadrantes para detectar predominancia regional
    - Cálculo de simetría bilateral (importante para detectar asimetrías motoras)
    - Medidas de dispersión y centro de masa del movimiento
    
    Estas características permiten distinguir patrones de movimiento normales
    (simétricos, distribuidos uniformemente) vs. anormales (asimétricos,
    concentrados en regiones específicas).
    """
    
    def __init__(self, n_quadrants: int = 4):
        """
        Inicializa el extractor espacial con parámetros optimizados.
        
        Args:
            n_quadrants: Número de cuadrantes para análisis
                         (4 = grid 2x2 para análisis bilateral,
                          9 = grid 3x3 para análisis más detallado)
        """
        self.n_quadrants = n_quadrants
        # Features: actividad por cuadrante + simetría + área + dispersión + centro
        self.n_features_ = n_quadrants + 4
        
    def fit(self, X, y=None):
        """No requiere ajuste."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extrae características espaciales de los videos.
        
        Args:
            X: Array de videos (n_samples, n_frames, height, width, channels)
            
        Returns:
            Array de características (n_samples, n_features)
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_features_))
        
        print(f"Extrayendo características espaciales de {n_samples} videos...")
        
        for i in tqdm(range(n_samples), desc="Spatial Features"):
            features[i] = self._extract_video_features(X[i])
            
        return features
    
    def _extract_video_features(self, video: np.ndarray) -> np.ndarray:
        """
        Extrae características espaciales de un solo video.
        
        Args:
            video: Video (n_frames, height, width, channels)
            
        Returns:
            Vector de características
        """
        n_frames, height, width, _ = video.shape
        
        # Calcular diferencia acumulada de frames (mapa de movimiento)
        movement_map = np.zeros((height, width), dtype=np.float32)
        
        for i in range(1, n_frames):
            diff = np.abs(
                video[i].astype(np.float32) - video[i-1].astype(np.float32)
            )
            movement_map += np.mean(diff, axis=2)
        
        movement_map /= (n_frames - 1)
        
        # Actividad por cuadrante
        quadrant_activity = self._compute_quadrant_activity(movement_map)
        
        # Simetría lateral (izquierda vs derecha)
        symmetry = self._compute_symmetry(movement_map)
        
        # Área de movimiento significativo
        area = self._compute_movement_area(movement_map)
        
        # Dispersión del movimiento
        dispersion = self._compute_dispersion(movement_map)
        
        # Centro de movimiento normalizado
        center_x, center_y = self._compute_movement_center(movement_map)
        
        features = list(quadrant_activity) + [symmetry, area, dispersion, center_x]
        
        return np.array(features)
    
    def _compute_quadrant_activity(self, movement_map: np.ndarray) -> list:
        """
        Calcula la actividad de movimiento por cuadrante.
        
        Args:
            movement_map: Mapa de movimiento acumulado
            
        Returns:
            Lista de actividad por cuadrante
        """
        h, w = movement_map.shape
        activities = []
        
        if self.n_quadrants == 4:
            # 2x2 grid
            quadrants = [
                movement_map[:h//2, :w//2],      # Superior izquierdo
                movement_map[:h//2, w//2:],      # Superior derecho
                movement_map[h//2:, :w//2],      # Inferior izquierdo
                movement_map[h//2:, w//2:]       # Inferior derecho
            ]
        else:
            # 3x3 grid
            h3, w3 = h//3, w//3
            quadrants = []
            for i in range(3):
                for j in range(3):
                    q = movement_map[i*h3:(i+1)*h3, j*w3:(j+1)*w3]
                    quadrants.append(q)
        
        for q in quadrants:
            activities.append(np.mean(q))
            
        # Normalizar
        total = sum(activities) + 1e-8
        activities = [a / total for a in activities]
        
        return activities
    
    def _compute_symmetry(self, movement_map: np.ndarray) -> float:
        """
        Calcula el índice de simetría lateral.
        
        Args:
            movement_map: Mapa de movimiento
            
        Returns:
            Índice de simetría (0 = asimétrico, 1 = simétrico)
        """
        h, w = movement_map.shape
        left = movement_map[:, :w//2]
        right = np.fliplr(movement_map[:, w//2:])
        
        # Ajustar tamaños si son diferentes
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        # Correlación entre lados
        correlation = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        
        # Manejar NaN
        if np.isnan(correlation):
            correlation = 0.0
            
        return correlation
    
    def _compute_movement_area(self, movement_map: np.ndarray) -> float:
        """
        Calcula el área con movimiento significativo.
        
        Args:
            movement_map: Mapa de movimiento
            
        Returns:
            Proporción del área con movimiento
        """
        threshold = np.mean(movement_map) + np.std(movement_map)
        significant = movement_map > threshold
        return np.mean(significant)
    
    def _compute_dispersion(self, movement_map: np.ndarray) -> float:
        """
        Calcula la dispersión espacial del movimiento.
        
        Args:
            movement_map: Mapa de movimiento
            
        Returns:
            Índice de dispersión (0 = concentrado, 1 = disperso)
        """
        # Calcular entropía espacial normalizada
        flat = movement_map.flatten()
        flat = flat / (flat.sum() + 1e-8)
        flat = flat[flat > 0]
        
        if len(flat) == 0:
            return 0.0
        
        entropy = -np.sum(flat * np.log2(flat + 1e-10))
        max_entropy = np.log2(len(movement_map.flatten()))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_movement_center(self, movement_map: np.ndarray) -> tuple:
        """
        Calcula el centro de masa del movimiento normalizado.
        
        Args:
            movement_map: Mapa de movimiento
            
        Returns:
            Coordenadas (x, y) normalizadas del centro
        """
        h, w = movement_map.shape
        
        # Crear grids de coordenadas
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        total_mass = movement_map.sum() + 1e-8
        
        cx = (x_coords * movement_map).sum() / total_mass
        cy = (y_coords * movement_map).sum() / total_mass
        
        # Normalizar a [-1, 1]
        cx_norm = (cx / w) * 2 - 1
        cy_norm = (cy / h) * 2 - 1
        
        return cx_norm, cy_norm
    
    def get_feature_names(self) -> list:
        """Retorna nombres de las características."""
        names = [f'spatial_quadrant_{i}' for i in range(self.n_quadrants)]
        names.extend([
            'spatial_symmetry',
            'spatial_area',
            'spatial_dispersion',
            'spatial_center_x'
        ])
        return names


if __name__ == "__main__":
    # Prueba del módulo
    import sys
    sys.path.append('..')
    from data.loader import DataLoader
    
    loader = DataLoader()
    data, targets = loader.load_sample("100_50_50")
    
    extractor = SpatialFeatureExtractor()
    features = extractor.fit_transform(data[:5])
    
    print(f"\nFeatures extraídas: {features.shape}")
    print(f"Nombres: {extractor.get_feature_names()}")
    print(f"Ejemplo:\n{features[0]}")
