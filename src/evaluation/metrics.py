"""

Sistema de ML para Detección de Espasticidad en Movimientos Infantiles
Universitat Oberta de Catalunya
Autor: Máximo Fernández Riera

Sistema completo de evaluación y métricas para modelos de ML.

Este módulo proporciona un framework comprehensivo para evaluar el rendimiento
de los modelos de clasificación multiclase en el problema de detección de
espasticidad, incluyendo métricas clínicas específicas y análisis detallado.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, cohen_kappa_score, matthews_corrcoef,
    log_loss, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluador comprehensivo para modelos de clasificación multiclase.
    
    Proporciona evaluación exhaustiva con métricas estándar y específicas
    del dominio clínico, visualizaciones y análisis comparativo.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Inicializa el evaluador.
        
        Args:
            class_names: Nombres de las clases para mejor interpretabilidad
        """
        self.class_names = class_names
        self.evaluation_results = {}
        
    def evaluate_model(self, model: Any, X_test: np.ndarray, 
                       y_test: np.ndarray, model_name: str) -> Dict:
        """
        Evaluación completa de un modelo individual.
        
        Calcula todas las métricas relevantes para evaluación clínica
        y diagnóstico de ML, incluyendo métricas específicas para
        detección temprana de espasticidad.
        
        Args:
            model: Modelo entrenado
            X_test: Características de test
            y_test: Etiquetas verdaderas
            model_name: Nombre del modelo para identificación
            
        Returns:
            Diccionario exhaustivo con todas las métricas calculadas
        """
        print(f"\n📊 Evaluando modelo: {model_name}")
        print("=" * 50)
        
        # Predicciones básicas
        y_pred = model.predict(X_test)
        
        # Obtener probabilidades si están disponibles
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            # Para modelos sin probabilidades, usar decision_function
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X_test)
                # Convertir a probabilidades pseudo con softmax
                y_proba = self._softmax(decision)
            else:
                y_proba = None
        
        # Métricas básicas de clasificación
        basic_metrics = self._calculate_basic_metrics(y_test, y_pred)
        
        # Métricas por clase
        per_class_metrics = self._calculate_per_class_metrics(y_test, y_pred)
        
        # Métricas probabilísticas
        if y_proba is not None:
            prob_metrics = self._calculate_probabilistic_metrics(y_test, y_proba)
        else:
            prob_metrics = {}
        
        # Métricas clínicas específicas
        clinical_metrics = self._calculate_clinical_metrics(y_test, y_pred, y_proba)
        
        # Análisis de errores
        error_analysis = self._analyze_errors(y_test, y_pred, y_proba)
        
        # Matriz de confusión detallada
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Compilar todos los resultados
        results = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'probabilistic_metrics': prob_metrics,
            'clinical_metrics': clinical_metrics,
            'error_analysis': error_analysis,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        # Almacenar para comparación posterior
        self.evaluation_results[model_name] = results
        
        # Imprimir resumen
        self._print_evaluation_summary(results)
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas básicas de clasificación.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            
        Returns:
            Diccionario con métricas básicas
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas individuales por cada clase.
        
        Esencial para identificar clases problemáticas y entender
        el rendimiento diferencial del modelo.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            
        Returns:
            Diccionario con métricas por clase
        """
        unique_classes = np.unique(y_true)
        per_class = {}
        
        for class_id in unique_classes:
            # Crear máscaras binarias para la clase actual
            y_true_binary = (y_true == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)
            
            # Calcular métricas para esta clase
            class_name = self.class_names[class_id] if self.class_names else f"Class_{class_id}"
            
            # Calcular TP, TN, FP, FN
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Métricas derivadas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[class_name] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'support': np.sum(y_true == class_id),
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        return per_class
    
    def _calculate_probabilistic_metrics(self, y_true: np.ndarray, 
                                        y_proba: np.ndarray) -> Dict:
        """
        Calcula métricas basadas en probabilidades.
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            
        Returns:
            Diccionario con métricas probabilísticas
        """
        n_classes = y_proba.shape[1]
        
        # Binarizar etiquetas para métricas multiclase
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))
        
        # Log loss (cross-entropy)
        logloss = log_loss(y_true, y_proba)
        
        # AUC-ROC multiclase
        try:
            # One-vs-Rest AUC
            auc_ovr = roc_auc_score(y_true_binarized, y_proba, multi_class='ovr')
            # One-vs-One AUC
            auc_ovo = roc_auc_score(y_true_binarized, y_proba, multi_class='ovo')
        except:
            auc_ovr = auc_ovo = 0.0
        
        # Brier score (calibración)
        brier_score = np.mean(np.sum((y_proba - y_true_binarized) ** 2, axis=1))
        
        # Entropía de las predicciones (incertidumbre)
        entropy = -np.mean(np.sum(y_proba * np.log(y_proba + 1e-10), axis=1))
        
        # Confianza promedio
        max_proba = np.max(y_proba, axis=1)
        avg_confidence = np.mean(max_proba)
        
        return {
            'log_loss': logloss,
            'auc_roc_ovr': auc_ovr,
            'auc_roc_ovo': auc_ovo,
            'brier_score': brier_score,
            'prediction_entropy': entropy,
            'average_confidence': avg_confidence,
            'confidence_std': np.std(max_proba)
        }
    
    def _calculate_clinical_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray]) -> Dict:
        """
        Calcula métricas específicas para el contexto clínico.
        
        Estas métricas son especialmente relevantes para el diagnóstico
        temprano de espasticidad en neonatos, donde los falsos negativos
        pueden tener consecuencias graves.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_proba: Probabilidades opcionales
            
        Returns:
            Diccionario con métricas clínicas
        """
        # Definir clases de riesgo (simplificado: clases 0-2 bajo, 3-5 medio, 6-7 alto)
        risk_levels = self._assign_risk_levels(y_true)
        risk_pred = self._assign_risk_levels(y_pred)
        
        # Sensibilidad para detección de alto riesgo
        high_risk_mask = risk_levels == 2  # Alto riesgo
        if np.sum(high_risk_mask) > 0:
            high_risk_sensitivity = np.sum((risk_pred == 2) & high_risk_mask) / np.sum(high_risk_mask)
        else:
            high_risk_sensitivity = 0.0
        
        # Valor predictivo negativo (VPN) - crítico en screening
        true_negatives = np.sum((risk_levels == 0) & (risk_pred == 0))
        false_negatives = np.sum((risk_levels > 0) & (risk_pred == 0))
        npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        
        # Tasa de falsos negativos en casos críticos
        critical_cases = risk_levels == 2
        if np.sum(critical_cases) > 0:
            critical_fn_rate = np.sum((risk_pred < 2) & critical_cases) / np.sum(critical_cases)
        else:
            critical_fn_rate = 0.0
        
        # Análisis de decisiones inciertas (probabilidad cercana a 0.5)
        uncertain_decisions = 0
        if y_proba is not None:
            max_proba = np.max(y_proba, axis=1)
            uncertain_decisions = np.sum((max_proba > 0.4) & (max_proba < 0.6)) / len(max_proba)
        
        # Número necesario para diagnosticar (NND)
        # Inverso del valor predictivo positivo para casos de alto riesgo
        high_risk_predictions = risk_pred == 2
        if np.sum(high_risk_predictions) > 0:
            high_risk_ppv = np.sum((risk_levels == 2) & high_risk_predictions) / np.sum(high_risk_predictions)
            nnd = 1 / high_risk_ppv if high_risk_ppv > 0 else float('inf')
        else:
            nnd = float('inf')
        
        return {
            'high_risk_sensitivity': high_risk_sensitivity,
            'negative_predictive_value': npv,
            'critical_false_negative_rate': critical_fn_rate,
            'uncertain_decision_rate': uncertain_decisions,
            'number_needed_to_diagnose': nnd,
            'risk_classification_accuracy': accuracy_score(risk_levels, risk_pred)
        }
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray]) -> Dict:
        """
        Análisis detallado de patrones de error.
        
        Identifica patrones sistemáticos en los errores del modelo
        para guiar mejoras futuras.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_proba: Probabilidades opcionales
            
        Returns:
            Diccionario con análisis de errores
        """
        errors_mask = y_true != y_pred
        n_errors = np.sum(errors_mask)
        n_total = len(y_true)
        
        # Matriz de confusión de errores
        error_pairs = []
        if n_errors > 0:
            error_true = y_true[errors_mask]
            error_pred = y_pred[errors_mask]
            
            for true_class, pred_class in zip(error_true, error_pred):
                error_pairs.append((true_class, pred_class))
            
            # Contar pares de errores más frecuentes
            from collections import Counter
            error_counter = Counter(error_pairs)
            top_errors = error_counter.most_common(5)
        else:
            top_errors = []
        
        # Análisis de confianza en errores vs aciertos
        confidence_analysis = {}
        if y_proba is not None:
            max_proba = np.max(y_proba, axis=1)
            confidence_analysis = {
                'avg_confidence_correct': np.mean(max_proba[~errors_mask]) if np.sum(~errors_mask) > 0 else 0,
                'avg_confidence_errors': np.mean(max_proba[errors_mask]) if np.sum(errors_mask) > 0 else 0,
                'confidence_gap': 0
            }
            confidence_analysis['confidence_gap'] = (
                confidence_analysis['avg_confidence_correct'] - 
                confidence_analysis['avg_confidence_errors']
            )
        
        # Distribución de errores por clase
        error_distribution = {}
        for class_id in np.unique(y_true):
            class_mask = y_true == class_id
            class_errors = np.sum(errors_mask & class_mask)
            class_total = np.sum(class_mask)
            error_distribution[f"class_{class_id}"] = {
                'error_count': class_errors,
                'total': class_total,
                'error_rate': class_errors / class_total if class_total > 0 else 0
            }
        
        return {
            'total_errors': n_errors,
            'error_rate': n_errors / n_total,
            'top_confusion_pairs': top_errors,
            'confidence_analysis': confidence_analysis,
            'error_distribution': error_distribution
        }
    
    def compare_models(self, models_dict: Dict[str, Any],
                       X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compara múltiples modelos sistemáticamente.
        
        Args:
            models_dict: Diccionario {nombre: modelo}
            X_test: Características de test
            y_test: Etiquetas de test
            
        Returns:
            DataFrame con comparación de métricas
        """
        print("\n🔍 Comparación de Modelos")
        print("=" * 70)
        
        comparison_data = []
        
        for model_name, model in models_dict.items():
            results = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Compilar métricas principales para comparación
            row = {
                'Model': model_name,
                'Accuracy': results['basic_metrics']['accuracy'],
                'Precision': results['basic_metrics']['precision_macro'],
                'Recall': results['basic_metrics']['recall_macro'],
                'F1-Score': results['basic_metrics']['f1_macro'],
                'Cohen Kappa': results['basic_metrics']['cohen_kappa']
            }
            
            # Añadir métricas probabilísticas si están disponibles
            if results['probabilistic_metrics']:
                row.update({
                    'AUC-ROC': results['probabilistic_metrics']['auc_roc_ovr'],
                    'Log Loss': results['probabilistic_metrics']['log_loss'],
                    'Avg Confidence': results['probabilistic_metrics']['average_confidence']
                })
            
            # Añadir métricas clínicas
            row.update({
                'High Risk Sens': results['clinical_metrics']['high_risk_sensitivity'],
                'NPV': results['clinical_metrics']['negative_predictive_value'],
                'Critical FN Rate': results['clinical_metrics']['critical_false_negative_rate']
            })
            
            comparison_data.append(row)
        
        # Crear DataFrame de comparación
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordenar por métrica principal (AUC-ROC o Accuracy)
        if 'AUC-ROC' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Añadir ranking
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        # Formatear para mejor visualización
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Rank']
        comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
        
        return comparison_df
    
    def _assign_risk_levels(self, labels: np.ndarray) -> np.ndarray:
        """
        Asigna niveles de riesgo basados en las clases.
        
        Mapeo simplificado para demostración:
        - Clases 0-2: Riesgo bajo (movimientos normales)
        - Clases 3-5: Riesgo medio (anomalías leves)
        - Clases 6-7: Riesgo alto (patrones espásticos)
        
        Args:
            labels: Etiquetas de clase
            
        Returns:
            Array con niveles de riesgo (0=bajo, 1=medio, 2=alto)
        """
        risk = np.zeros_like(labels)
        risk[(labels >= 3) & (labels <= 5)] = 1
        risk[labels >= 6] = 2
        return risk
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica función softmax para convertir scores en probabilidades.
        
        Args:
            x: Scores o decision functions
            
        Returns:
            Probabilidades normalizadas
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _print_evaluation_summary(self, results: Dict):
        """
        Imprime un resumen conciso de la evaluación.
        
        Args:
            results: Diccionario de resultados de evaluación
        """
        print(f"\n📈 Resumen de {results['model_name']}:")
        print("-" * 40)
        
        # Métricas básicas
        basic = results['basic_metrics']
        print(f"Accuracy:    {basic['accuracy']:.4f}")
        print(f"Precision:   {basic['precision_macro']:.4f}")
        print(f"Recall:      {basic['recall_macro']:.4f}")
        print(f"F1-Score:    {basic['f1_macro']:.4f}")
        
        # Métricas probabilísticas si están disponibles
        if results['probabilistic_metrics']:
            prob = results['probabilistic_metrics']
            print(f"AUC-ROC:     {prob['auc_roc_ovr']:.4f}")
            print(f"Log Loss:    {prob['log_loss']:.4f}")
        
        # Métricas clínicas
        clinical = results['clinical_metrics']
        print(f"\n🏥 Métricas Clínicas:")
        print(f"High Risk Sensitivity: {clinical['high_risk_sensitivity']:.4f}")
        print(f"NPV:                  {clinical['negative_predictive_value']:.4f}")
        print(f"Critical FN Rate:     {clinical['critical_false_negative_rate']:.4f}")
        
        # Análisis de errores
        errors = results['error_analysis']
        print(f"\n❌ Análisis de Errores:")
        print(f"Total errores: {errors['total_errors']} ({errors['error_rate']:.2%})")
        if errors['confidence_analysis']:
            print(f"Gap confianza: {errors['confidence_analysis']['confidence_gap']:.3f}")
    
    def generate_report(self, output_path: str = "evaluation_report.txt"):
        """
        Genera un informe completo de evaluación.
        
        Args:
            output_path: Ruta para guardar el informe
        """
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("INFORME COMPLETO DE EVALUACIÓN DE MODELOS\n")
            f.write("=" * 70 + "\n\n")
            
            for model_name, results in self.evaluation_results.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * len(model_name) + "\n\n")
                
                # Escribir todas las métricas
                f.write("Métricas Básicas:\n")
                for metric, value in results['basic_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                
                f.write("\nMétricas por Clase:\n")
                for class_name, metrics in results['per_class_metrics'].items():
                    f.write(f"  {class_name}:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"    {metric}: {value:.4f}\n")
                        else:
                            f.write(f"    {metric}: {value}\n")
                
                f.write("\n" + "=" * 70 + "\n")
        
        print(f"\n📝 Informe guardado en: {output_path}")


if __name__ == "__main__":
    # Código de prueba
    print("Módulo de Evaluación de Modelos")
    print("=" * 50)
    
    # Simular datos y modelo
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(42)
    
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 8, 100)
    X_test = np.random.randn(30, 20)
    y_test = np.random.randint(0, 8, 30)
    
    # Entrenar modelo de prueba
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, "RandomForest_Test")
    
    print("\n✅ Módulo de evaluación funcionando correctamente")
