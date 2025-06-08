"""
PromptGen Enterprise - Sistema de Configuración Centralizada
===========================================================

Sistema de configuración empresarial que maneja:
- Configuraciones de modelos y parámetros
- Variables de entorno y secretos
- Configuraciones de monitoreo y alertas
- Configuraciones de deployment y escalabilidad

Autor: Senior DevOps Engineer
Versión: 2.0.0 Enterprise
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuración de modelo empresarial"""
    name: str
    model_id: str
    model_type: str  # 'causal-lm', 'seq2seq-lm', 'text-classification'
    parameters: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    description: str = ""
    enabled: bool = True
    priority: int = 1  # 1=alta, 2=media, 3=baja

@dataclass
class MonitoringConfig:
    """Configuración de monitoreo empresarial"""
    enabled: bool = True
    metrics_retention_hours: int = 24
    alert_cooldown_minutes: int = 5
    performance_thresholds: Dict[str, float] = None
    business_metrics_enabled: bool = True
    export_enabled: bool = True
    dashboard_refresh_seconds: int = 30
    
    def __post_init__(self):
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                'cpu_usage_warning': 70.0,
                'cpu_usage_critical': 85.0,
                'memory_usage_warning': 75.0,
                'memory_usage_critical': 90.0,
                'error_rate_warning': 2.0,
                'error_rate_critical': 5.0,
                'response_time_warning': 3.0,
                'response_time_critical': 10.0
            }

@dataclass
class APIConfig:
    """Configuración de API empresarial"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = None
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    request_timeout_seconds: int = 300
    max_request_size_mb: int = 10
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:3001"
            ]

@dataclass
class QualityConfig:
    """Configuración de análisis de calidad"""
    enabled_metrics: List[str] = None
    quality_thresholds: Dict[str, float] = None
    improvement_target_default: float = 85.0
    max_iterations_default: int = 5
    min_quality_improvement: float = 2.0
    
    def __post_init__(self):
        if self.enabled_metrics is None:
            self.enabled_metrics = [
                'completeness',
                'clarity', 
                'specificity',
                'structure',
                'coherence',
                'actionability'
            ]
        
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'excellent': 90.0,
                'good': 75.0,
                'acceptable': 60.0,
                'poor': 40.0
            }

class EnterpriseConfig:
    """Gestor de configuración empresarial centralizada"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "enterprise_config.yaml"
        self.config_dir = Path(__file__).parent / "config"
        self.config_path = self.config_dir / self.config_file
        
        # Configuraciones por defecto
        self._default_models = self._get_default_models()
        self.monitoring = MonitoringConfig()
        self.api = APIConfig()
        self.quality = QualityConfig()
        
        # Variables de entorno
        self.environment = os.getenv("PROMPTGEN_ENV", "development")
        self.debug = os.getenv("PROMPTGEN_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("PROMPTGEN_LOG_LEVEL", "INFO")
        
        # Cargar configuración desde archivo
        self.load_config()
        
        logger.info(f"✅ Configuración empresarial cargada - Entorno: {self.environment}")
    
    def _get_default_models(self) -> Dict[str, ModelConfig]:
        """Obtiene configuraciones de modelos por defecto"""
        return {
            "gpt2": ModelConfig(
                name="GPT-2",
                model_id="gpt2",
                model_type="causal-lm",
                parameters="124M",
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                description="Modelo GPT-2 base de OpenAI para generación de texto",
                priority=1
            ),
            "distilgpt2": ModelConfig(
                name="DistilGPT-2",
                model_id="distilgpt2",
                model_type="causal-lm", 
                parameters="82M",
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                description="Versión destilada y optimizada de GPT-2",
                priority=2
            ),
            "gpt-neo-125m": ModelConfig(
                name="GPT-Neo 125M",
                model_id="EleutherAI/gpt-neo-125M",
                model_type="causal-lm",
                parameters="125M",
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                description="Modelo GPT-Neo de EleutherAI",
                priority=2
            ),
            "t5-small": ModelConfig(
                name="T5-Small",
                model_id="t5-small",
                model_type="seq2seq-lm",
                parameters="60M",
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                description="Modelo T5-Small de Google para tareas seq2seq",
                priority=3
            )
        }
    
    def load_config(self):
        """Carga configuración desde archivo YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Cargar configuraciones específicas
                if 'monitoring' in config_data:
                    self.monitoring = MonitoringConfig(**config_data['monitoring'])
                
                if 'api' in config_data:
                    self.api = APIConfig(**config_data['api'])
                
                if 'quality' in config_data:
                    self.quality = QualityConfig(**config_data['quality'])
                
                # Cargar modelos personalizados
                if 'models' in config_data:
                    for model_key, model_data in config_data['models'].items():
                        self._default_models[model_key] = ModelConfig(**model_data)
                
                logger.info(f"✅ Configuración cargada desde {self.config_path}")
            else:
                logger.info("📝 Usando configuración por defecto - creando archivo de configuración")
                self.save_config()
                
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            logger.info("🔄 Usando configuración por defecto")
    
    def save_config(self):
        """Guarda configuración actual a archivo YAML"""
        try:
            # Crear directorio si no existe
            self.config_dir.mkdir(exist_ok=True)
            
            config_data = {
                'environment': self.environment,
                'debug': self.debug,
                'log_level': self.log_level,
                'monitoring': asdict(self.monitoring),
                'api': asdict(self.api),
                'quality': asdict(self.quality),
                'models': {
                    key: asdict(model) for key, model in self._default_models.items()
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Configuración guardada en {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando configuración: {e}")
    
    def get_model_config(self, model_key: str) -> Optional[ModelConfig]:
        """Obtiene configuración de un modelo específico"""
        return self._default_models.get(model_key)
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """Obtiene solo los modelos habilitados"""
        return {
            key: config for key, config in self._default_models.items()
            if config.enabled
        }
    
    def get_models_by_priority(self, priority: int = None) -> Dict[str, ModelConfig]:
        """Obtiene modelos filtrados por prioridad"""
        if priority is None:
            return self.get_enabled_models()
        
        return {
            key: config for key, config in self._default_models.items()
            if config.enabled and config.priority == priority
        }
    
    def update_model_config(self, model_key: str, **kwargs):
        """Actualiza configuración de un modelo"""
        if model_key in self._default_models:
            model_config = self._default_models[model_key]
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            logger.info(f"✅ Configuración del modelo {model_key} actualizada")
        else:
            logger.warning(f"⚠️ Modelo {model_key} no encontrado")
    
    def add_model_config(self, model_key: str, model_config: ModelConfig):
        """Añade nueva configuración de modelo"""
        self._default_models[model_key] = model_config
        logger.info(f"✅ Nuevo modelo {model_key} añadido a la configuración")
    
    def remove_model_config(self, model_key: str):
        """Elimina configuración de modelo"""
        if model_key in self._default_models:
            del self._default_models[model_key]
            logger.info(f"✅ Modelo {model_key} eliminado de la configuración")
        else:
            logger.warning(f"⚠️ Modelo {model_key} no encontrado")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del entorno"""
        base_config = {
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level
        }
        
        if self.environment == "production":
            base_config.update({
                'api_workers': 4,
                'api_reload': False,
                'monitoring_enabled': True,
                'rate_limiting_enabled': True,
                'cors_strict': True
            })
        elif self.environment == "staging":
            base_config.update({
                'api_workers': 2,
                'api_reload': False,
                'monitoring_enabled': True,
                'rate_limiting_enabled': True,
                'cors_strict': False
            })
        else:  # development
            base_config.update({
                'api_workers': 1,
                'api_reload': True,
                'monitoring_enabled': True,
                'rate_limiting_enabled': False,
                'cors_strict': False
            })
        
        return base_config
    
    def validate_config(self) -> List[str]:
        """Valida la configuración actual y retorna lista de errores"""
        errors = []
        
        # Validar modelos
        if not self._default_models:
            errors.append("No hay modelos configurados")
        
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            errors.append("No hay modelos habilitados")
        
        # Validar configuración de API
        if self.api.port < 1 or self.api.port > 65535:
            errors.append(f"Puerto de API inválido: {self.api.port}")
        
        if self.api.workers < 1:
            errors.append(f"Número de workers inválido: {self.api.workers}")
        
        # Validar configuración de monitoreo
        if self.monitoring.metrics_retention_hours < 1:
            errors.append("Retención de métricas debe ser al menos 1 hora")
        
        # Validar configuración de calidad
        if self.quality.improvement_target_default < 0 or self.quality.improvement_target_default > 100:
            errors.append("Objetivo de mejora debe estar entre 0 y 100")
        
        if self.quality.max_iterations_default < 1:
            errors.append("Máximo de iteraciones debe ser al menos 1")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la configuración actual"""
        enabled_models = self.get_enabled_models()
        
        return {
            'environment': self.environment,
            'debug_mode': self.debug,
            'total_models': len(self._default_models),
            'enabled_models': len(enabled_models),
            'model_list': list(enabled_models.keys()),
            'api_config': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'cors_origins_count': len(self.api.cors_origins)
            },
            'monitoring_enabled': self.monitoring.enabled,
            'quality_metrics_count': len(self.quality.enabled_metrics),
            'config_file': str(self.config_path),
            'config_valid': len(self.validate_config()) == 0
        }

# Instancia global de configuración
_enterprise_config = None

def get_enterprise_config() -> EnterpriseConfig:
    """Obtiene la instancia global de configuración empresarial"""
    global _enterprise_config
    if _enterprise_config is None:
        _enterprise_config = EnterpriseConfig()
    return _enterprise_config

def reload_enterprise_config():
    """Recarga la configuración empresarial"""
    global _enterprise_config
    _enterprise_config = None
    return get_enterprise_config()

# Funciones de conveniencia
def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """Función de conveniencia para obtener configuración de modelo"""
    return get_enterprise_config().get_model_config(model_key)

def get_enabled_models() -> Dict[str, ModelConfig]:
    """Función de conveniencia para obtener modelos habilitados"""
    return get_enterprise_config().get_enabled_models()

def get_monitoring_config() -> MonitoringConfig:
    """Función de conveniencia para obtener configuración de monitoreo"""
    return get_enterprise_config().monitoring

def get_api_config() -> APIConfig:
    """Función de conveniencia para obtener configuración de API"""
    return get_enterprise_config().api

def get_quality_config() -> QualityConfig:
    """Función de conveniencia para obtener configuración de calidad"""
    return get_enterprise_config().quality 