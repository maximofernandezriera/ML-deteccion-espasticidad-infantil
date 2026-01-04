"""

Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Módulo de división estratificada de datos.

"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Tuple
import yaml


class DataSplitter:
    """División estratificada de datos para train/validation/test."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el divisor de datos.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        split_config = config['data']['split']
        self.train_ratio = split_config['train_ratio']
        self.val_ratio = split_config['val_ratio']
        self.test_ratio = split_config['test_ratio']
        self.random_state = config['project']['random_state']
        
        # Validar proporciones
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, \
            f"Las proporciones deben sumar 1.0, suman {total}"
        
    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Divide datos en train/val/test manteniendo estratificación.
        
        Args:
            X: Datos de entrada
            y: Etiquetas
            
        Returns:
            Diccionario con splits: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        # Primero separar test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_ratio, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Luego separar val de train
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            stratify=y_temp,
            random_state=self.random_state
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        self._print_split_info(splits, y)
        
        return splits
    
    def split_indices(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Retorna solo los índices de la división.
        
        Args:
            y: Etiquetas
            
        Returns:
            Diccionario con índices de cada split
        """
        indices = np.arange(len(y))
        
        # Primero separar test
        idx_temp, idx_test, y_temp, _ = train_test_split(
            indices, y,
            test_size=self.test_ratio,
            stratify=y,
            random_state=self.random_state
        )
        
        # Luego separar val de train
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        idx_train, idx_val, _, _ = train_test_split(
            idx_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        return {
            'train': idx_train,
            'val': idx_val,
            'test': idx_test
        }
    
    def get_cv_folds(self, n_splits: int = 5) -> StratifiedKFold:
        """
        Retorna objeto de cross-validation estratificado.
        
        Args:
            n_splits: Número de folds
            
        Returns:
            StratifiedKFold configurado
        """
        return StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
    
    def _print_split_info(self, splits: Dict, y_original: np.ndarray) -> None:
        """Imprime información sobre la división realizada."""
        print("\n" + "="*50)
        print("DIVISIÓN DE DATOS")
        print("="*50)
        
        for name, (X, y) in splits.items():
            pct = 100 * len(y) / len(y_original)
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n{name.upper()}: {len(y)} muestras ({pct:.1f}%)")
            print(f"  Distribución de clases:")
            for cls, cnt in zip(unique, counts):
                print(f"    Clase {cls}: {cnt}")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    # Prueba del módulo
    from loader import DataLoader
    
    loader = DataLoader()
    data, targets = loader.load_sample("100_50_50")
    
    splitter = DataSplitter()
    splits = splitter.split(data, targets)
    
    print("\nVerificación de shapes:")
    for name, (X, y) in splits.items():
        print(f"  {name}: X={X.shape}, y={y.shape}")
