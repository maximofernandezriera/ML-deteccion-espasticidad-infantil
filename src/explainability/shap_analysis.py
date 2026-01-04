"""

Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Análisis de interpretabilidad con SHAP para modelos de ML.

Este módulo proporciona análisis de interpretabilidad usando valores SHAP
(SHapley Additive exPlanations) para entender las decisiones del modelo
en el contexto de detección de espasticidad infantil.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    Analizador SHAP para interpretabilidad de modelos.
    
    Proporciona explicaciones locales y globales de las predicciones del modelo,
    crucial para la confianza clínica en el diagnóstico automatizado.
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, 
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None):
        """
        Inicializa el analizador SHAP.
        
        Args:
            model: Modelo entrenado a explicar
            X_train: Datos de entrenamiento para background
            feature_names: Nombres de las características
            class_names: Nombres de las clases
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        self.class_names = class_names or [f"Class_{i}" for i in range(8)]
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Determinar tipo de explainer según el modelo
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """
        Inicializa el explainer SHAP apropiado según el tipo de modelo.
        
        Selección automática del explainer más eficiente:
        - TreeExplainer: Para modelos basados en árboles (RF, XGBoost)
        - LinearExplainer: Para modelos lineales (Logistic Regression)
        - KernelExplainer: Para cualquier modelo (más lento pero universal)
        """
        model_type = type(self.model).__name__
        
        print(f"🔮 Inicializando SHAP Explainer para {model_type}...")
        
        try:
            if 'Forest' in model_type or 'Tree' in model_type:
                # Random Forest, Decision Trees
                self.explainer = shap.TreeExplainer(self.model)
                print("   Usando TreeExplainer (exacto y rápido)")
                
            elif 'XGB' in model_type or 'xgboost' in str(type(self.model)).lower():
                # XGBoost
                self.explainer = shap.TreeExplainer(self.model)
                print("   Usando TreeExplainer optimizado para XGBoost")
                
            elif 'Logistic' in model_type or 'Linear' in model_type:
                # Logistic Regression, Linear models
                self.explainer = shap.LinearExplainer(
                    self.model, 
                    self.X_train
                )
                print("   Usando LinearExplainer")
                
            elif 'SVC' in model_type or 'SVM' in model_type:
                # Support Vector Machines
                # Usar KernelExplainer con función de predicción
                background = shap.sample(self.X_train, 100)  # Submuestra para eficiencia
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                else:
                    predict_fn = self.model.decision_function
                
                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    background
                )
                print("   Usando KernelExplainer (aproximación)")
                
            else:
                # Modelo genérico - usar KernelExplainer
                background = shap.sample(self.X_train, 100)
                predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    background
                )
                print("   Usando KernelExplainer genérico")
                
        except Exception as e:
            print(f"   ⚠️ Error inicializando explainer: {e}")
            print("   Intentando con KernelExplainer genérico...")
            
            # Fallback a KernelExplainer
            background = shap.sample(self.X_train, 50)
            predict_fn = self.model.predict if not hasattr(self.model, 'predict_proba') else self.model.predict_proba
            self.explainer = shap.KernelExplainer(predict_fn, background)
    
    def explain_predictions(self, X_explain: np.ndarray, 
                           sample_size: Optional[int] = None) -> Dict:
        """
        Calcula valores SHAP para las predicciones.
        
        Args:
            X_explain: Datos a explicar
            sample_size: Número de muestras a explicar (None = todas)
            
        Returns:
            Diccionario con valores SHAP y análisis
        """
        if sample_size:
            X_explain = X_explain[:sample_size]
        
        print(f"\n📊 Calculando valores SHAP para {X_explain.shape[0]} muestras...")
        
        # Calcular valores SHAP
        try:
            if isinstance(self.explainer, shap.KernelExplainer):
                # KernelExplainer puede ser lento, limitar muestras
                n_samples = min(100, X_explain.shape[0])
                self.shap_values = self.explainer.shap_values(X_explain[:n_samples])
                print(f"   Valores calculados para {n_samples} muestras (KernelExplainer)")
            else:
                self.shap_values = self.explainer.shap_values(X_explain)
                print(f"   Valores calculados para todas las muestras")
            
            # Obtener valor esperado
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
            
        except Exception as e:
            print(f"   Error calculando SHAP values: {e}")
            return {}
        
        # Analizar resultados
        analysis = self._analyze_shap_values()
        
        return {
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'analysis': analysis
        }
    
    def _analyze_shap_values(self) -> Dict:
        """
        Analiza los valores SHAP calculados.
        
        Returns:
            Diccionario con análisis de importancia y patrones
        """
        if self.shap_values is None:
            return {}
        
        # Manejar caso multiclase
        if isinstance(self.shap_values, list):
            # Promediar importancia absoluta entre clases
            avg_abs_shap = np.mean([
                np.abs(sv).mean(axis=0) for sv in self.shap_values
            ], axis=0)
            
            # Análisis por clase
            class_importance = {}
            for i, sv in enumerate(self.shap_values):
                if i < len(self.class_names):
                    class_name = self.class_names[i]
                    importance = np.abs(sv).mean(axis=0)
                    class_importance[class_name] = dict(zip(
                        self.feature_names,
                        importance / importance.sum()
                    ))
        else:
            # Caso binario o regresión
            avg_abs_shap = np.abs(self.shap_values).mean(axis=0)
            class_importance = {}
        
        # Normalizar importancia global
        global_importance = avg_abs_shap / avg_abs_shap.sum()
        
        # Crear ranking de características
        feature_ranking = sorted(
            zip(self.feature_names, global_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Identificar características dominantes
        cumsum = np.cumsum(sorted(global_importance, reverse=True))
        n_dominant = np.argmax(cumsum >= 0.8) + 1  # Features para 80% de importancia
        
        # Análisis de interacciones (si hay suficientes features importantes)
        interaction_analysis = self._analyze_feature_interactions()
        
        return {
            'global_importance': dict(zip(self.feature_names, global_importance)),
            'feature_ranking': feature_ranking,
            'top_10_features': feature_ranking[:10],
            'n_dominant_features': n_dominant,
            'class_specific_importance': class_importance,
            'interaction_analysis': interaction_analysis
        }
    
    def _analyze_feature_interactions(self) -> Dict:
        """
        Analiza interacciones entre características basado en SHAP.
        
        Returns:
            Diccionario con análisis de interacciones
        """
        if self.shap_values is None:
            return {}
        
        # Tomar primera clase o valores directos para análisis
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Calcular correlación entre valores SHAP de diferentes features
        # Alta correlación sugiere interacción
        n_features = shap_vals.shape[1]
        
        if n_features > 100:  # Limitar para eficiencia
            # Tomar solo top features
            importance = np.abs(shap_vals).mean(axis=0)
            top_indices = np.argsort(importance)[-20:]
            shap_vals = shap_vals[:, top_indices]
            feature_subset = [self.feature_names[i] for i in top_indices]
        else:
            feature_subset = self.feature_names
        
        # Calcular matriz de correlación
        shap_df = pd.DataFrame(shap_vals, columns=feature_subset[:shap_vals.shape[1]])
        correlation_matrix = shap_df.corr()
        
        # Identificar pares con alta correlación
        high_interactions = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Umbral de correlación significativa
                    high_interactions.append({
                        'feature_1': correlation_matrix.index[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Ordenar por correlación absoluta
        high_interactions = sorted(
            high_interactions,
            key=lambda x: abs(x['correlation']),
            reverse=True
        )[:10]  # Top 10 interacciones
        
        return {
            'n_interactions_found': len(high_interactions),
            'top_interactions': high_interactions,
            'interaction_strength': np.mean([abs(x['correlation']) for x in high_interactions]) if high_interactions else 0
        }
    
    def explain_single_prediction(self, X_single: np.ndarray, 
                                 true_class: Optional[int] = None) -> Dict:
        """
        Explicación detallada de una predicción individual.
        
        Útil para casos clínicos específicos donde se necesita
        entender por qué el modelo tomó una decisión particular.
        
        Args:
            X_single: Muestra individual (1, n_features)
            true_class: Clase verdadera (opcional)
            
        Returns:
            Diccionario con explicación detallada
        """
        # Asegurar formato correcto
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        # Obtener predicción del modelo
        prediction = self.model.predict(X_single)[0]
        
        # Probabilidades si están disponibles
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_single)[0]
            confidence = np.max(probabilities)
        else:
            probabilities = None
            confidence = None
        
        # Calcular SHAP values para esta muestra
        shap_values_single = self.explainer.shap_values(X_single)
        
        # Para multiclase, tomar valores de la clase predicha
        if isinstance(shap_values_single, list):
            shap_for_prediction = shap_values_single[prediction][0]
            
            # También obtener SHAP para clase verdadera si se proporciona
            if true_class is not None and true_class != prediction:
                shap_for_true = shap_values_single[true_class][0]
            else:
                shap_for_true = None
        else:
            shap_for_prediction = shap_values_single[0]
            shap_for_true = None
        
        # Identificar contribuciones principales
        contributions = sorted(
            zip(self.feature_names, shap_for_prediction, X_single[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Separar contribuciones positivas y negativas
        positive_contributions = [(f, s, v) for f, s, v in contributions if s > 0][:5]
        negative_contributions = [(f, s, v) for f, s, v in contributions if s < 0][:5]
        
        # Análisis de decisión
        decision_analysis = {
            'predicted_class': self.class_names[prediction] if prediction < len(self.class_names) else f"Class_{prediction}",
            'confidence': confidence,
            'probabilities': dict(zip(self.class_names[:len(probabilities)], probabilities)) if probabilities is not None else None
        }
        
        if true_class is not None:
            decision_analysis['true_class'] = self.class_names[true_class] if true_class < len(self.class_names) else f"Class_{true_class}"
            decision_analysis['correct'] = (prediction == true_class)
        
        # Compilar explicación
        explanation = {
            'decision': decision_analysis,
            'top_positive_factors': [
                {
                    'feature': feat,
                    'contribution': contrib,
                    'value': val,
                    'impact': 'Aumenta probabilidad'
                }
                for feat, contrib, val in positive_contributions
            ],
            'top_negative_factors': [
                {
                    'feature': feat,
                    'contribution': contrib,
                    'value': val,
                    'impact': 'Disminuye probabilidad'
                }
                for feat, contrib, val in negative_contributions
            ],
            'total_positive_contribution': sum(s for _, s, _ in positive_contributions),
            'total_negative_contribution': sum(s for _, s, _ in negative_contributions)
        }
        
        # Si hay discrepancia entre predicción y verdad
        if shap_for_true is not None:
            true_contributions = sorted(
                zip(self.feature_names, shap_for_true - shap_for_prediction),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            explanation['misclassification_factors'] = [
                {
                    'feature': feat,
                    'difference': diff,
                    'interpretation': 'Favorecería clase correcta' if diff > 0 else 'Desfavorece clase correcta'
                }
                for feat, diff in true_contributions
            ]
        
        return explanation
    
    def generate_feature_importance_report(self) -> pd.DataFrame:
        """
        Genera un reporte de importancia de características.
        
        Returns:
            DataFrame con análisis de importancia
        """
        if self.shap_values is None:
            print("⚠️ Primero debe ejecutar explain_predictions()")
            return pd.DataFrame()
        
        # Calcular importancias
        if isinstance(self.shap_values, list):
            # Multiclase
            importance_data = []
            
            # Importancia global
            global_importance = np.mean([
                np.abs(sv).mean(axis=0) for sv in self.shap_values
            ], axis=0)
            
            # Por clase
            for i, sv in enumerate(self.shap_values):
                if i < len(self.class_names):
                    class_importance = np.abs(sv).mean(axis=0)
                    for j, feat in enumerate(self.feature_names):
                        importance_data.append({
                            'Feature': feat,
                            'Class': self.class_names[i],
                            'Importance': class_importance[j],
                            'Global_Importance': global_importance[j]
                        })
        else:
            # Binario/Regresión
            importance = np.abs(self.shap_values).mean(axis=0)
            importance_data = [
                {
                    'Feature': feat,
                    'Class': 'All',
                    'Importance': imp,
                    'Global_Importance': imp
                }
                for feat, imp in zip(self.feature_names, importance)
            ]
        
        # Crear DataFrame
        df = pd.DataFrame(importance_data)
        
        # Añadir ranking
        global_ranking = df.groupby('Feature')['Global_Importance'].mean().rank(ascending=False)
        df['Global_Rank'] = df['Feature'].map(global_ranking)
        
        # Normalizar importancias
        df['Normalized_Importance'] = df.groupby('Class')['Importance'].transform(
            lambda x: x / x.sum()
        )
        
        # Ordenar por importancia global
        df = df.sort_values(['Global_Rank', 'Class'])
        
        return df
    
    def create_waterfall_plot(self, sample_index: int, class_index: Optional[int] = None,
                             max_features: int = 10) -> Dict:
        """
        Crea datos para un gráfico waterfall de contribuciones SHAP.
        
        Args:
            sample_index: Índice de la muestra a explicar
            class_index: Índice de la clase (None = clase predicha)
            max_features: Número máximo de características a mostrar
            
        Returns:
            Diccionario con datos para el gráfico
        """
        if self.shap_values is None:
            return {}
        
        # Obtener valores SHAP para la muestra
        if isinstance(self.shap_values, list):
            if class_index is None:
                # Usar clase predicha
                prediction = self.model.predict(self.X_train[sample_index:sample_index+1])[0]
                class_index = prediction
            
            shap_sample = self.shap_values[class_index][sample_index]
            base_value = self.expected_value[class_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        else:
            shap_sample = self.shap_values[sample_index]
            base_value = self.expected_value
        
        # Ordenar por importancia absoluta
        indices = np.argsort(np.abs(shap_sample))[-max_features:]
        
        # Preparar datos para waterfall
        waterfall_data = {
            'base_value': base_value,
            'features': [self.feature_names[i] for i in indices],
            'values': [self.X_train[sample_index, i] for i in indices],
            'shap_values': [shap_sample[i] for i in indices],
            'cumulative': []
        }
        
        # Calcular valores acumulativos
        cumsum = base_value
        for shap_val in waterfall_data['shap_values']:
            cumsum += shap_val
            waterfall_data['cumulative'].append(cumsum)
        
        waterfall_data['final_prediction'] = cumsum
        
        return waterfall_data
    
    def get_clinical_insights(self) -> Dict:
        """
        Genera insights específicos para el contexto clínico.
        
        Returns:
            Diccionario con interpretaciones clínicas de los SHAP values
        """
        if self.shap_values is None:
            return {}
        
        analysis = self._analyze_shap_values()
        
        if not analysis:
            return {}
        
        # Identificar patrones clínicamente relevantes
        top_features = analysis['top_10_features']
        
        clinical_insights = {
            'movement_patterns': [],
            'temporal_features': [],
            'spatial_features': [],
            'risk_indicators': []
        }
        
        # Clasificar features por tipo (basado en nombres)
        for feature, importance in top_features:
            feature_lower = feature.lower()
            
            if 'velocity' in feature_lower or 'acceleration' in feature_lower:
                clinical_insights['movement_patterns'].append({
                    'feature': feature,
                    'importance': importance,
                    'interpretation': 'Patrón de movimiento crítico para diagnóstico'
                })
            elif 'temporal' in feature_lower or 'time' in feature_lower or 'freq' in feature_lower:
                clinical_insights['temporal_features'].append({
                    'feature': feature,
                    'importance': importance,
                    'interpretation': 'Característica temporal relevante'
                })
            elif 'spatial' in feature_lower or 'position' in feature_lower or 'symmetry' in feature_lower:
                clinical_insights['spatial_features'].append({
                    'feature': feature,
                    'importance': importance,
                    'interpretation': 'Patrón espacial significativo'
                })
            
            # Identificar indicadores de riesgo
            if importance > 0.1:  # Alta importancia
                clinical_insights['risk_indicators'].append({
                    'feature': feature,
                    'importance': importance,
                    'risk_level': 'Alto' if importance > 0.15 else 'Medio'
                })
        
        # Resumen ejecutivo
        clinical_insights['summary'] = {
            'n_critical_features': len([f for f, i in top_features if i > 0.1]),
            'dominant_pattern_type': 'movement' if clinical_insights['movement_patterns'] else 'temporal',
            'complexity': 'Alta' if analysis['n_dominant_features'] > 10 else 'Media' if analysis['n_dominant_features'] > 5 else 'Baja',
            'interpretability': 'Clara' if analysis['n_dominant_features'] < 5 else 'Moderada' if analysis['n_dominant_features'] < 10 else 'Compleja'
        }
        
        return clinical_insights


if __name__ == "__main__":
    # Código de prueba
    print("Módulo de Análisis SHAP")
    print("=" * 50)
    
    # Simular datos y modelo
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(42)
    
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 8, 100)
    X_test = np.random.randn(20, 20)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Crear analizador SHAP
    feature_names = [f"Feature_{i}" for i in range(20)]
    class_names = [f"Movement_Type_{i}" for i in range(8)]
    
    analyzer = SHAPAnalyzer(model, X_train, feature_names, class_names)
    
    # Explicar predicciones
    results = analyzer.explain_predictions(X_test, sample_size=5)
    
    if results:
        print("\n✅ Análisis SHAP completado")
        print(f"Top 3 características más importantes:")
        for feat, imp in results['analysis']['top_10_features'][:3]:
            print(f"  - {feat}: {imp:.4f}")
    
    # Explicación individual
    explanation = analyzer.explain_single_prediction(X_test[0], true_class=3)
    print(f"\n🔍 Explicación de predicción individual:")
    print(f"  Clase predicha: {explanation['decision']['predicted_class']}")
    if explanation['top_positive_factors']:
        print(f"  Factor principal: {explanation['top_positive_factors'][0]['feature']}")
    
    print("\n✅ Módulo SHAP funcionando correctamente")
