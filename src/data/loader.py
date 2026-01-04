"""

Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Módulo de carga de datos del dataset de movimientos infantiles.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import yaml


class DataLoader:
    """Cargador de datos del dataset de movimientos infantiles."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el cargador de datos.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.data_path = Path(self.config['data']['raw_path'])
        self.random_state = self.config['project']['random_state']
        
    def load_sample(self, sample_name: str = "100_50_50") -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga un sample específico del dataset.
        
        Args:
            sample_name: Nombre del sample (ej: "100_50_50")
            
        Returns:
            Tuple con (datos, etiquetas)
        """
        data_file = self.data_path / f"data_{sample_name}.npz"
        target_file = self.data_path / f"target_{sample_name}.npz"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {data_file}")
        if not target_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {target_file}")
        
        data = np.load(data_file)['arr_0']
        targets = np.load(target_file)['arr_0']
        
        self._validate_data(data, targets)
        
        print(f"Datos cargados exitosamente:")
        print(f"  - Shape datos: {data.shape}")
        print(f"  - Shape targets: {targets.shape}")
        print(f"  - Clases únicas: {np.unique(targets)}")
        
        return data, targets
    
    def _validate_data(self, data: np.ndarray, targets: np.ndarray) -> None:
        """
        Valida la integridad de los datos cargados.
        
        Args:
            data: Array de datos
            targets: Array de etiquetas
            
        Raises:
            AssertionError: Si los datos no pasan validación
        """
        assert data.shape[0] == targets.shape[0], \
            f"Mismatch en número de muestras: {data.shape[0]} vs {targets.shape[0]}"
        assert len(data.shape) == 5, \
            f"Se esperaba tensor 5D, se obtuvo {len(data.shape)}D"
        assert not np.isnan(data).any(), \
            "Se encontraron valores NaN en los datos"
        assert data.dtype == np.uint8, \
            f"Tipo de datos inesperado: {data.dtype}"
        
    def get_statistics(self, data: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Retorna estadísticas básicas del dataset.
        
        Args:
            data: Array de datos
            targets: Array de etiquetas
            
        Returns:
            Diccionario con estadísticas
        """
        unique, counts = np.unique(targets, return_counts=True)
        
        return {
            'n_samples': data.shape[0],
            'n_frames': data.shape[1],
            'height': data.shape[2],
            'width': data.shape[3],
            'channels': data.shape[4],
            'dtype': str(data.dtype),
            'memory_mb': round(data.nbytes / (1024**2), 2),
            'n_classes': len(unique),
            'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
            'pixel_stats': {
                'min': int(np.min(data)),
                'max': int(np.max(data)),
                'mean': round(float(np.mean(data)), 2),
                'std': round(float(np.std(data)), 2)
            }
        }
    
    def print_summary(self, data: np.ndarray, targets: np.ndarray) -> None:
        """Imprime un resumen formateado del dataset."""
        stats = self.get_statistics(data, targets)
        
        print("\n" + "="*60)
        print("RESUMEN DEL DATASET")
        print("="*60)
        print(f"\nDimensiones:")
        print(f"  - Muestras: {stats['n_samples']}")
        print(f"  - Frames/video: {stats['n_frames']}")
        print(f"  - Resolución: {stats['height']}x{stats['width']}")
        print(f"  - Canales: {stats['channels']} (RGB)")
        print(f"  - Memoria: {stats['memory_mb']} MB")
        
        print(f"\nClases ({stats['n_classes']} categorías):")
        for cls, count in stats['class_distribution'].items():
            pct = 100 * count / stats['n_samples']
            print(f"  - Clase {cls}: {count} muestras ({pct:.1f}%)")
        
        print(f"\nEstadísticas de píxeles:")
        for key, val in stats['pixel_stats'].items():
            print(f"  - {key}: {val}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Prueba del módulo
    loader = DataLoader()
    data, targets = loader.load_sample("100_50_50")
    loader.print_summary(data, targets)
