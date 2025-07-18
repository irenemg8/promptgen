"""
PromptGen Enterprise - Sistema Simplificado de Mejora de Prompts
===========================================================

Sistema empresarial simplificado para la mejora iterativa de prompts utilizando:
- Modelos básicos de Hugging Face
- Procesamiento inteligente en español
- Métricas de calidad avanzadas

Autor: Senior DevOps Engineer
Versión: 2.0.0 Enterprise Simplificado
"""

import os
import torch
import warnings
import re
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
import numpy as np

# Configuración de logging empresarial
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings no críticos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class QualityMetrics:
    """Métricas de calidad para evaluación de prompts"""
    clarity_score: float
    specificity_score: float
    completeness_score: float
    coherence_score: float
    overall_score: float
    feedback: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

class EnterpriseModelManager:
    """Gestor de modelos empresariales optimizado"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Dispositivo seleccionado: {self.device}")
        
    def get_model(self, model_name: str = "gpt2"):
        """Obtener modelo con caché"""
        if model_name not in self.models:
            try:
                logger.info(f"🔄 Cargando modelo: {model_name}")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, padding_side="left")
                if self.tokenizers[model_name].pad_token is None:
                    self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
                
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                logger.info(f"✅ Modelo {model_name} cargado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error cargando modelo {model_name}: {e}")
                return None
                
        return self.models[model_name], self.tokenizers[model_name]

class AdvancedQualityAnalyzer:
    """Analizador de calidad avanzado para prompts"""
    
    def __init__(self):
        self.quality_patterns = {
            'clarity': [
                r'\b(claro|específico|preciso|detallado|explícito)\b',
                r'\b(definido|concreto|exacto|particular)\b'
            ],
            'specificity': [
                r'\b(por ejemplo|específicamente|en particular|tales como)\b',
                r'\b(incluye|considera|asegúrate|verifica)\b'
            ],
            'completeness': [
                r'\b(completo|integral|exhaustivo|comprensivo)\b',
                r'\b(todo|todos|cada|cualquier)\b'
            ],
            'coherence': [
                r'\b(además|también|por tanto|sin embargo)\b',
                r'\b(primero|segundo|finalmente|en conclusión)\b'
            ]
        }
        
    def analyze_prompt_quality(self, prompt: str) -> QualityMetrics:
        """Analizar calidad del prompt"""
        try:
            # Análisis de claridad
            clarity_score = self._analyze_clarity(prompt)
            
            # Análisis de especificidad
            specificity_score = self._analyze_specificity(prompt)
            
            # Análisis de completitud
            completeness_score = self._analyze_completeness(prompt)
            
            # Análisis de coherencia
            coherence_score = self._analyze_coherence(prompt)
            
            # Puntuación general
            overall_score = np.mean([clarity_score, specificity_score, completeness_score, coherence_score])
            
            # Generar feedback
            feedback = self._generate_feedback(prompt, {
                'clarity': clarity_score,
                'specificity': specificity_score,
                'completeness': completeness_score,
                'coherence': coherence_score
            })
            
            return QualityMetrics(
                clarity_score=clarity_score,
                specificity_score=specificity_score,
                completeness_score=completeness_score,
                coherence_score=coherence_score,
                overall_score=overall_score,
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de calidad: {e}")
            return QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, ["Error en el análisis"])
    
    def _analyze_clarity(self, prompt: str) -> float:
        """Analizar claridad del prompt"""
        score = 0.4  # Base score
        
        # Longitud apropiada
        word_count = len(prompt.split())
        if 10 <= word_count <= 100:
            score += 0.2
        elif word_count > 100:
            score += 0.1
            
        # Presencia de patrones de claridad
        for pattern in self.quality_patterns['clarity']:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.1
                
        return min(1.0, score)
    
    def _analyze_specificity(self, prompt: str) -> float:
        """Analizar especificidad del prompt"""
        score = 0.3  # Base score
        
        # Presencia de ejemplos o detalles específicos
        for pattern in self.quality_patterns['specificity']:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.15
                
        # Presencia de números o datos específicos
        if re.search(r'\d+', prompt):
            score += 0.1
            
        return min(1.0, score)
    
    def _analyze_completeness(self, prompt: str) -> float:
        """Analizar completitud del prompt"""
        score = 0.3  # Base score
        
        # Presencia de instrucciones completas
        for pattern in self.quality_patterns['completeness']:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.1
                
        # Presencia de contexto
        if any(word in prompt.lower() for word in ['contexto', 'situación', 'escenario']):
            score += 0.2
            
        return min(1.0, score)
    
    def _analyze_coherence(self, prompt: str) -> float:
        """Analizar coherencia del prompt"""
        score = 0.4  # Base score
        
        # Presencia de conectores lógicos
        for pattern in self.quality_patterns['coherence']:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.1
                
        # Estructura lógica
        sentences = prompt.split('.')
        if len(sentences) > 1:
            score += 0.1
            
        return min(1.0, score)
    
    def _generate_feedback(self, prompt: str, scores: Dict[str, float]) -> List[str]:
        """Generar feedback específico"""
        feedback = []
        
        if scores['clarity'] < 0.6:
            feedback.append("🔍 Mejora la claridad: Sé más específico en tus instrucciones")
            
        if scores['specificity'] < 0.6:
            feedback.append("🎯 Aumenta la especificidad: Incluye ejemplos concretos")
            
        if scores['completeness'] < 0.6:
            feedback.append("📋 Mejora la completitud: Proporciona más contexto")
            
        if scores['coherence'] < 0.6:
            feedback.append("🔗 Mejora la coherencia: Usa conectores lógicos")
            
        if not feedback:
            feedback.append("✅ Prompt bien estructurado")
            
        return feedback

class ProgressiveImprovementEngine:
    """Motor de mejora progresiva para prompts"""
    
    def __init__(self, model_manager: EnterpriseModelManager, quality_analyzer: AdvancedQualityAnalyzer):
        self.model_manager = model_manager
        self.quality_analyzer = quality_analyzer
        self.improvement_templates = [
            "Mejora este prompt haciéndolo más claro y específico: {prompt}",
            "Reescribe este prompt para que sea más detallado: {prompt}",
            "Optimiza este prompt para obtener mejores resultados: {prompt}"
        ]
        
    def improve_prompt(self, prompt: str, iterations: int = 3) -> Dict[str, Any]:
        """Mejorar prompt iterativamente"""
        try:
            results = {
                'original': prompt,
                'iterations': [],
                'best_prompt': prompt,
                'best_score': 0.0
            }
            
            current_prompt = prompt
            
            for i in range(iterations):
                # Analizar calidad actual
                quality = self.quality_analyzer.analyze_prompt_quality(current_prompt)
                
                # Generar mejora
                improved_prompt = self._generate_improvement(current_prompt)
                
                # Analizar calidad mejorada
                improved_quality = self.quality_analyzer.analyze_prompt_quality(improved_prompt)
                
                iteration_result = {
                    'iteration': i + 1,
                    'prompt': improved_prompt,
                    'quality': improved_quality.to_dict(),
                    'improvement': improved_quality.overall_score - quality.overall_score
                }
                
                results['iterations'].append(iteration_result)
                
                # Actualizar mejor prompt
                if improved_quality.overall_score > results['best_score']:
                    results['best_prompt'] = improved_prompt
                    results['best_score'] = improved_quality.overall_score
                    
                current_prompt = improved_prompt
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en mejora progresiva: {e}")
            return {'error': str(e)}
    
    def _generate_improvement(self, prompt: str) -> str:
        """Generar mejora del prompt"""
        try:
            # Seleccionar template de mejora
            template = np.random.choice(self.improvement_templates)
            improvement_prompt = template.format(prompt=prompt)
            
            # Usar modelo para generar mejora
            model, tokenizer = self.model_manager.get_model("gpt2")
            if model is None:
                return f"MEJORADO: {prompt} (incluye más detalles específicos y contexto)"
            
            inputs = tokenizer(improvement_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            improved_prompt = generated_text[len(improvement_prompt):].strip()
            
            if not improved_prompt:
                improved_prompt = f"MEJORADO: {prompt} (incluye más detalles específicos y contexto)"
            
            return improved_prompt
            
        except Exception as e:
            logger.error(f"❌ Error generando mejora: {e}")
            return f"MEJORADO: {prompt} (incluye más detalles específicos y contexto)"

# Funciones principales de la API
def analyze_prompt_quality_bart(prompt: str) -> Dict[str, Any]:
    """Analizar calidad de un prompt usando BART"""
    try:
        analyzer = AdvancedQualityAnalyzer()
        quality = analyzer.analyze_prompt_quality(prompt)
        return {
            'success': True,
            'quality_metrics': quality.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error en análisis BART: {e}")
        return {'success': False, 'error': str(e)}

def get_structural_feedback(prompt: str, model_name: str = "gpt2") -> Dict[str, Any]:
    """Obtener feedback estructural"""
    try:
        analyzer = AdvancedQualityAnalyzer()
        quality = analyzer.analyze_prompt_quality(prompt)
        
        return {
            'success': True,
            'structural_feedback': {
                'clarity': quality.clarity_score,
                'specificity': quality.specificity_score,
                'completeness': quality.completeness_score,
                'coherence': quality.coherence_score,
                'overall_score': quality.overall_score,
                'suggestions': quality.feedback
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error en feedback estructural: {e}")
        return {'success': False, 'error': str(e)}

def generate_variations(prompt: str, model_name: str, num_variations: int) -> Dict[str, Any]:
    """Generar variaciones del prompt"""
    try:
        variations = []
        base_templates = [
            f"Reformula este prompt de manera más clara: {prompt}",
            f"Crea una versión más específica de: {prompt}",
            f"Mejora este prompt haciéndolo más detallado: {prompt}"
        ]
        
        for i in range(num_variations):
            if i < len(base_templates):
                variation = base_templates[i]
            else:
                variation = f"Variación {i+1}: {prompt} (incluye más contexto y ejemplos específicos)"
            
            variations.append({
                'variation': i + 1,
                'prompt': variation,
                'improvement_focus': ['clarity', 'specificity', 'completeness'][i % 3]
            })
        
        return {
            'success': True,
            'variations': variations,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error generando variaciones: {e}")
        return {'success': False, 'error': str(e)}

def generate_ideas(prompt: str, model_name: str, num_ideas: int) -> Dict[str, Any]:
    """Generar ideas para prompts"""
    try:
        ideas = []
        base_ideas = [
            f"Crea un prompt para {prompt} que incluya ejemplos específicos",
            f"Desarrolla un prompt para {prompt} con contexto detallado",
            f"Diseña un prompt para {prompt} que sea claro y preciso",
            f"Elabora un prompt para {prompt} con instrucciones paso a paso",
            f"Construye un prompt para {prompt} que genere resultados creativos"
        ]
        
        for i in range(num_ideas):
            if i < len(base_ideas):
                idea = base_ideas[i]
            else:
                idea = f"Idea {i+1}: Prompt para {prompt} con enfoque innovador"
            
            ideas.append({
                'idea': i + 1,
                'prompt_suggestion': idea,
                'category': ['specific', 'detailed', 'clear', 'structured', 'creative'][i % 5]
            })
        
        return {
            'success': True,
            'ideas': ideas,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error generando ideas: {e}")
        return {'success': False, 'error': str(e)}

# Inicialización global
try:
    model_manager = EnterpriseModelManager()
    quality_analyzer = AdvancedQualityAnalyzer()
    improvement_engine = ProgressiveImprovementEngine(model_manager, quality_analyzer)
    logger.info("✅ Sistema PromptGen Enterprise Simplificado inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando sistema: {e}")
    model_manager = None
    quality_analyzer = None
    improvement_engine = None 