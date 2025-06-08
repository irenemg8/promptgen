"""
PromptGen Enterprise - Sistema de Monitoreo y Observabilidad
===========================================================

Sistema empresarial de monitoreo para tracking de rendimiento,
métricas de negocio, alertas y observabilidad completa.

Autor: Senior DevOps Engineer
Versión: 2.0.0 Enterprise
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import asyncio
from pathlib import Path

# Configuración de logging empresarial
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del sistema"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    active_sessions: int
    total_requests: int
    error_rate: float
    model_load_time: float
    quality_analysis_time: float

@dataclass
class BusinessMetrics:
    """Métricas de negocio empresariales"""
    timestamp: datetime
    total_prompts_processed: int
    successful_improvements: int
    average_quality_improvement: float
    user_satisfaction_score: float
    model_usage_distribution: Dict[str, int]
    peak_usage_hours: List[int]
    conversion_rate: float

@dataclass
class AlertConfig:
    """Configuración de alertas empresariales"""
    metric_name: str
    threshold: float
    comparison: str  # 'greater', 'less', 'equal'
    severity: str    # 'low', 'medium', 'high', 'critical'
    cooldown_minutes: int = 5

class EnterpriseMonitoringSystem:
    """Sistema de monitoreo empresarial para PromptGen"""
    
    def __init__(self, metrics_retention_hours: int = 24):
        self.metrics_retention_hours = metrics_retention_hours
        self.performance_metrics = deque(maxlen=1000)
        self.business_metrics = deque(maxlen=1000)
        self.alert_history = deque(maxlen=500)
        self.active_alerts = {}
        
        # Configuraciones de alertas por defecto
        self.alert_configs = [
            AlertConfig("cpu_usage", 80.0, "greater", "high"),
            AlertConfig("memory_usage", 85.0, "greater", "high"),
            AlertConfig("error_rate", 5.0, "greater", "medium"),
            AlertConfig("response_time", 5.0, "greater", "medium"),
            AlertConfig("model_load_time", 30.0, "greater", "low"),
        ]
        
        # Contadores de sesión
        self.session_counters = defaultdict(int)
        self.request_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        
        # Métricas de negocio
        self.business_counters = {
            'total_prompts': 0,
            'successful_improvements': 0,
            'total_quality_improvement': 0.0,
            'model_usage': defaultdict(int)
        }
        
        # Iniciar thread de monitoreo
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🔍 Sistema de monitoreo empresarial iniciado")
    
    def record_request(self, endpoint: str, response_time: float, success: bool = True):
        """Registra una petición HTTP"""
        current_hour = datetime.now().hour
        self.request_counters[current_hour] += 1
        
        if not success:
            self.error_counters[current_hour] += 1
    
    def record_model_usage(self, model_name: str, load_time: float):
        """Registra uso de modelo"""
        self.business_counters['model_usage'][model_name] += 1
        
        # Actualizar métricas de rendimiento
        self._update_performance_metrics(model_load_time=load_time)
    
    def record_prompt_improvement(self, original_quality: float, improved_quality: float):
        """Registra mejora de prompt"""
        self.business_counters['total_prompts'] += 1
        
        if improved_quality > original_quality:
            self.business_counters['successful_improvements'] += 1
            improvement = improved_quality - original_quality
            self.business_counters['total_quality_improvement'] += improvement
    
    def record_session_activity(self, session_id: str):
        """Registra actividad de sesión"""
        current_hour = datetime.now().hour
        self.session_counters[current_hour] += 1
    
    def _update_performance_metrics(self, response_time: float = 0, model_load_time: float = 0, quality_analysis_time: float = 0):
        """Actualiza métricas de rendimiento del sistema"""
        try:
            # Métricas del sistema
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calcular tasa de error
            current_hour = datetime.now().hour
            total_requests = self.request_counters[current_hour]
            total_errors = self.error_counters[current_hour]
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            # Sesiones activas (estimación basada en última hora)
            active_sessions = sum(self.session_counters[h] for h in range(max(0, current_hour-1), current_hour+1))
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                response_time=response_time,
                active_sessions=active_sessions,
                total_requests=total_requests,
                error_rate=error_rate,
                model_load_time=model_load_time,
                quality_analysis_time=quality_analysis_time
            )
            
            self.performance_metrics.append(metrics)
            self._check_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error actualizando métricas de rendimiento: {e}")
    
    def _update_business_metrics(self):
        """Actualiza métricas de negocio"""
        try:
            # Calcular métricas derivadas
            total_prompts = self.business_counters['total_prompts']
            successful_improvements = self.business_counters['successful_improvements']
            
            avg_quality_improvement = 0.0
            if successful_improvements > 0:
                avg_quality_improvement = self.business_counters['total_quality_improvement'] / successful_improvements
            
            conversion_rate = 0.0
            if total_prompts > 0:
                conversion_rate = (successful_improvements / total_prompts) * 100
            
            # Horas pico (últimas 24 horas)
            current_hour = datetime.now().hour
            peak_hours = []
            max_requests = 0
            
            for hour in range(24):
                requests = self.request_counters[hour]
                if requests > max_requests:
                    max_requests = requests
                    peak_hours = [hour]
                elif requests == max_requests and requests > 0:
                    peak_hours.append(hour)
            
            metrics = BusinessMetrics(
                timestamp=datetime.now(),
                total_prompts_processed=total_prompts,
                successful_improvements=successful_improvements,
                average_quality_improvement=avg_quality_improvement,
                user_satisfaction_score=85.0,  # Placeholder - se puede integrar con feedback real
                model_usage_distribution=dict(self.business_counters['model_usage']),
                peak_usage_hours=peak_hours,
                conversion_rate=conversion_rate
            )
            
            self.business_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error actualizando métricas de negocio: {e}")
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Verifica y dispara alertas según configuración"""
        for alert_config in self.alert_configs:
            try:
                metric_value = getattr(metrics, alert_config.metric_name, None)
                if metric_value is None:
                    continue
                
                should_alert = False
                if alert_config.comparison == "greater":
                    should_alert = metric_value > alert_config.threshold
                elif alert_config.comparison == "less":
                    should_alert = metric_value < alert_config.threshold
                elif alert_config.comparison == "equal":
                    should_alert = abs(metric_value - alert_config.threshold) < 0.1
                
                if should_alert:
                    self._trigger_alert(alert_config, metric_value, metrics.timestamp)
                    
            except Exception as e:
                logger.error(f"Error verificando alerta {alert_config.metric_name}: {e}")
    
    def _trigger_alert(self, config: AlertConfig, value: float, timestamp: datetime):
        """Dispara una alerta"""
        alert_key = f"{config.metric_name}_{config.severity}"
        
        # Verificar cooldown
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            if (timestamp - last_alert).total_seconds() < (config.cooldown_minutes * 60):
                return
        
        # Registrar alerta
        alert = {
            'timestamp': timestamp,
            'metric': config.metric_name,
            'value': value,
            'threshold': config.threshold,
            'severity': config.severity,
            'message': f"{config.metric_name} ({value:.2f}) excede umbral ({config.threshold})"
        }
        
        self.alert_history.append(alert)
        self.active_alerts[alert_key] = timestamp
        
        # Log según severidad
        if config.severity == "critical":
            logger.critical(f"🚨 ALERTA CRÍTICA: {alert['message']}")
        elif config.severity == "high":
            logger.error(f"⚠️ ALERTA ALTA: {alert['message']}")
        elif config.severity == "medium":
            logger.warning(f"⚡ ALERTA MEDIA: {alert['message']}")
        else:
            logger.info(f"ℹ️ ALERTA BAJA: {alert['message']}")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.monitoring_active:
            try:
                self._update_performance_metrics()
                self._update_business_metrics()
                self._cleanup_old_data()
                time.sleep(60)  # Actualizar cada minuto
                
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(60)
    
    def _cleanup_old_data(self):
        """Limpia datos antiguos según retención configurada"""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Limpiar métricas de rendimiento
        while self.performance_metrics and self.performance_metrics[0].timestamp < cutoff_time:
            self.performance_metrics.popleft()
        
        # Limpiar métricas de negocio
        while self.business_metrics and self.business_metrics[0].timestamp < cutoff_time:
            self.business_metrics.popleft()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard empresarial"""
        try:
            # Métricas actuales
            latest_perf = self.performance_metrics[-1] if self.performance_metrics else None
            latest_business = self.business_metrics[-1] if self.business_metrics else None
            
            # Alertas recientes (últimas 24 horas)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert['timestamp'] > datetime.now() - timedelta(hours=24)
            ]
            
            # Tendencias (últimas 6 horas)
            six_hours_ago = datetime.now() - timedelta(hours=6)
            recent_perf_metrics = [
                m for m in self.performance_metrics
                if m.timestamp > six_hours_ago
            ]
            
            return {
                'current_performance': asdict(latest_perf) if latest_perf else {},
                'current_business': asdict(latest_business) if latest_business else {},
                'recent_alerts': recent_alerts,
                'performance_trends': [asdict(m) for m in recent_perf_metrics],
                'system_health': self._calculate_system_health(),
                'uptime': self._calculate_uptime(),
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'critical'])
            }
            
        except Exception as e:
            logger.error(f"Error generando datos de dashboard: {e}")
            return {}
    
    def _calculate_system_health(self) -> str:
        """Calcula estado general del sistema"""
        if not self.performance_metrics:
            return "unknown"
        
        latest = self.performance_metrics[-1]
        
        # Criterios de salud
        if (latest.cpu_usage > 90 or 
            latest.memory_usage > 90 or 
            latest.error_rate > 10):
            return "critical"
        elif (latest.cpu_usage > 70 or 
              latest.memory_usage > 75 or 
              latest.error_rate > 5):
            return "warning"
        else:
            return "healthy"
    
    def _calculate_uptime(self) -> float:
        """Calcula uptime del sistema (placeholder)"""
        # En un sistema real, esto se calcularía basado en logs de inicio/parada
        return 99.9
    
    def export_metrics(self, filepath: str):
        """Exporta métricas a archivo JSON"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_metrics': [asdict(m) for m in self.performance_metrics],
                'business_metrics': [asdict(m) for m in self.business_metrics],
                'alert_history': list(self.alert_history),
                'system_counters': {
                    'session_counters': dict(self.session_counters),
                    'request_counters': dict(self.request_counters),
                    'error_counters': dict(self.error_counters),
                    'business_counters': {
                        'total_prompts': self.business_counters['total_prompts'],
                        'successful_improvements': self.business_counters['successful_improvements'],
                        'total_quality_improvement': self.business_counters['total_quality_improvement'],
                        'model_usage': dict(self.business_counters['model_usage'])
                    }
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"📊 Métricas exportadas a {filepath}")
            
        except Exception as e:
            logger.error(f"Error exportando métricas: {e}")
    
    def stop_monitoring(self):
        """Detiene el sistema de monitoreo"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("🔍 Sistema de monitoreo detenido")

# Instancia global del sistema de monitoreo
monitoring_system = EnterpriseMonitoringSystem()

def get_monitoring_system() -> EnterpriseMonitoringSystem:
    """Obtiene la instancia del sistema de monitoreo"""
    return monitoring_system