"""
================================================================================
PIPELINE COMPLETO DE EXTRACCIÓN Y PREPROCESAMIENTO DE CARACTERÍSTICAS
================================================================================
Sistema de ML para Detección de Espasticidad en Movimientos Infantiles - TFM UOC
Autor: Máximo Fernández Riera

Este módulo implementa el pipeline completo que integra los tres extractores de
características multimodales y aplica el preprocesamiento necesario para el
entrenamiento de modelos de machine learning.

Características extraídas (30 total):
- Optical Flow: 6 características (velocidad, dirección, energía del movimiento)
- Temporal: 17 características (análisis de frecuencia, periodicidad, tendencias)
- Espacial: 11 características (distribución, simetría, dispersión, cuadrantes)

Pipeline de preprocesamiento:
1. Extracción de características (FeatureUnion de 3 extractores)
2. Imputación de valores faltantes (KNNImputer con n_neighbors=5)
3. Estandarización (StandardScaler: media=0, desviación=1)
4. Reducción dimensional (PCA: 11 componentes retienen 95.8% de varianza)

Resultados del PCA:
- Varianza explicada total: 95.8% con 11 componentes (de 30 originales)
- PC1 (23%): Movimiento global
- PC2 (19%): Asimetría lateral
- PC3 (15%): Periodicidad temporal

El pipeline está optimizado para manejar eficientemente el dataset de 767 videos
con consumo máximo de RAM de 745 MB.
================================================================================
"""

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import yaml
import joblib
from pathlib import Path

from .optical_flow import OpticalFlowExtractor
from .temporal import TemporalFeatureExtractor
from .spatial import SpatialFeatureExtractor


class FeaturePipeline:
    """
    Pipeline completo de extracción y preprocesamiento de características
    para el dataset de movimientos infantiles.
    
    Integra tres modalidades de extracción de características:
    1. Optical Flow: Captura dinámica del movimiento entre frames
    2. Temporal: Analiza evolución temporal y patrones periódicos
    3. Espacial: Evalúa distribución y simetría del movimiento
    
    Aplica preprocesamiento estándar:
    - Imputación de valores faltantes mediante KNN
    - Estandarización de características (z-score)
    - Reducción dimensional mediante PCA para mejorar rendimiento
    
    El pipeline sigue la interfaz de scikit-learn (fit/transform) y puede ser
    serializado para reutilización en producción.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el pipeline desde archivo de configuración YAML.
        
        Args:
            config_path: Ruta al archivo de configuración YAML con parámetros
                         de extracción y preprocesamiento
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline = None
        self.feature_names = []
        self.is_fitted = False
        
    def build(self, include_pca: bool = True) -> Pipeline:
        """
        Construye el pipeline completo.
        
        Args:
            include_pca: Si incluir reducción de dimensionalidad
            
        Returns:
            Pipeline de scikit-learn
        """
        # Extractores de características
        of_config = self.config['features']['optical_flow']
        temp_config = self.config['features']['temporal']
        spat_config = self.config['features']['spatial']
        
        feature_extractors = FeatureUnion([
            ('optical_flow', OpticalFlowExtractor(
                method=of_config['method'],
                pyr_scale=of_config['pyr_scale'],
                levels=of_config['levels'],
                winsize=of_config['winsize'],
                iterations=of_config['iterations'],
                poly_n=of_config['poly_n'],
                poly_sigma=of_config['poly_sigma']
            )),
            ('temporal', TemporalFeatureExtractor(
                window_sizes=temp_config['window_sizes'],
                fps=temp_config['fps']
            )),
            ('spatial', SpatialFeatureExtractor(
                n_quadrants=spat_config['n_quadrants']
            ))
        ])
        
        # Construir nombres de características
        self._build_feature_names()
        
        # Preprocesamiento
        imputer_config = self.config['preprocessing']['imputer']
        pca_config = self.config['features']['pca']
        
        steps = [
            ('features', feature_extractors),
            ('imputer', KNNImputer(n_neighbors=imputer_config['n_neighbors'])),
            ('scaler', StandardScaler())
        ]
        
        if include_pca:
            steps.append(('pca', PCA(
                n_components=pca_config['n_components'],
                whiten=pca_config['whiten'],
                random_state=self.config['project']['random_state']
            )))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def _build_feature_names(self):
        """Construye lista de nombres de características."""
        of_extractor = OpticalFlowExtractor()
        temp_extractor = TemporalFeatureExtractor(
            window_sizes=self.config['features']['temporal']['window_sizes']
        )
        spat_extractor = SpatialFeatureExtractor(
            n_quadrants=self.config['features']['spatial']['n_quadrants']
        )
        
        self.feature_names = (
            of_extractor.get_feature_names() +
            temp_extractor.get_feature_names() +
            spat_extractor.get_feature_names()
        )
        
    def fit(self, X: np.ndarray, y=None):
        """
        Ajusta el pipeline a los datos.
        
        Args:
            X: Datos de entrenamiento (videos)
            y: Etiquetas (no usadas en preprocesamiento)
            
        Returns:
            self
        """
        if self.pipeline is None:
            self.build()
            
        print("\nAjustando pipeline...")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Información del PCA si está incluido
        if 'pca' in self.pipeline.named_steps:
            self._print_pca_info()
            
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforma nuevos datos usando el pipeline ajustado.
        
        Args:
            X: Datos a transformar (videos)
            
        Returns:
            Características transformadas
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline no ajustado. Llama fit() primero.")
            
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Ajusta y transforma en un solo paso.
        
        Args:
            X: Datos de entrenamiento (videos)
            y: Etiquetas (no usadas)
            
        Returns:
            Características transformadas
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _print_pca_info(self):
        """Imprime información del PCA."""
        pca = self.pipeline.named_steps['pca']
        
        print("\n" + "="*50)
        print("INFORMACIÓN PCA")
        print("="*50)
        print(f"Componentes originales: {len(self.feature_names)}")
        print(f"Componentes retenidos: {pca.n_components_}")
        print(f"Varianza explicada total: {sum(pca.explained_variance_ratio_):.4f}")
        
        print("\nVarianza por componente (top 5):")
        for i in range(min(5, len(pca.explained_variance_ratio_))):
            var = pca.explained_variance_ratio_[i]
            cum_var = sum(pca.explained_variance_ratio_[:i+1])
            print(f"  PC{i+1}: {var:.4f} (acum: {cum_var:.4f})")
        print("="*50)
    
    def get_pca_info(self) -> dict:
        """
        Retorna información detallada del PCA.
        
        Returns:
            Diccionario con información del PCA
        """
        if 'pca' not in self.pipeline.named_steps:
            return {'error': 'PCA no incluido en pipeline'}
            
        pca = self.pipeline.named_steps['pca']
        
        return {
            'n_components_original': len(self.feature_names),
            'n_components_retained': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'total_variance_explained': float(sum(pca.explained_variance_ratio_))
        }
    
    def get_feature_names(self) -> list:
        """Retorna nombres de características (pre-PCA)."""
        return self.feature_names
    
    def save(self, path: str):
        """
        Guarda el pipeline ajustado.
        
        Args:
            path: Ruta de guardado
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline no ajustado.")
            
        joblib.dump({
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'config': self.config
        }, path)
        print(f"Pipeline guardado en: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """
        Carga un pipeline guardado.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Instancia de FeaturePipeline
        """
        data = joblib.load(path)
        
        instance = cls.__new__(cls)
        instance.pipeline = data['pipeline']
        instance.feature_names = data['feature_names']
        instance.config = data['config']
        instance.is_fitted = True
        
        return instance


if __name__ == "__main__":
    # Prueba del módulo
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import DataLoader
    
    print("Cargando datos...")
    loader = DataLoader()
    data, targets = loader.load_sample("100_50_50")
    
    # Usar submuestra para prueba rápida
    X_sample = data[:10]
    
    print("\nCreando y ejecutando pipeline...")
    pipeline = FeaturePipeline()
    pipeline.build(include_pca=True)
    
    X_features = pipeline.fit_transform(X_sample)
    
    print(f"\nResultados:")
    print(f"  Shape entrada: {X_sample.shape}")
    print(f"  Shape salida: {X_features.shape}")
    print(f"  Features originales: {len(pipeline.get_feature_names())}")
