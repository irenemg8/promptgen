"""
PromptGen Enterprise - Dashboard de Monitoreo Web
================================================

Dashboard empresarial en tiempo real para visualización de:
- Métricas de rendimiento del sistema
- Métricas de negocio y KPIs
- Alertas y notificaciones
- Tendencias y análisis histórico

Autor: Senior DevOps Engineer
Versión: 2.0.0 Enterprise
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json

# Configuración de la página
st.set_page_config(
    page_title="PromptGen Enterprise Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de la API
API_BASE_URL = "http://localhost:8000"

def get_api_data(endpoint):
    """Obtiene datos de la API con manejo de errores"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error conectando con API: {e}")
        return None

def format_metric_value(value, metric_type="number"):
    """Formatea valores de métricas para visualización"""
    if value is None:
        return "N/A"
    
    if metric_type == "percentage":
        return f"{value:.1f}%"
    elif metric_type == "time":
        return f"{value:.2f}s"
    elif metric_type == "bytes":
        if value > 1024**3:
            return f"{value/1024**3:.1f} GB"
        elif value > 1024**2:
            return f"{value/1024**2:.1f} MB"
        else:
            return f"{value/1024:.1f} KB"
    else:
        return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)

def create_gauge_chart(value, title, max_value=100, color_ranges=None):
    """Crea un gráfico de gauge para métricas"""
    if color_ranges is None:
        color_ranges = [
            (0, 70, "green"),
            (70, 85, "yellow"), 
            (85, 100, "red")
        ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [r[0], r[1]], 'color': r[2]} 
                for r in color_ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Función principal del dashboard"""
    
    # Header del dashboard
    st.title("🚀 PromptGen Enterprise Dashboard")
    st.markdown("---")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        # Filtros de tiempo
        time_range = st.selectbox(
            "Rango de tiempo",
            ["Última hora", "Últimas 6 horas", "Últimas 24 horas", "Última semana"]
        )
        
        # Configuración de alertas
        st.subheader("🔔 Alertas")
        show_all_alerts = st.checkbox("Mostrar todas las alertas", value=False)
        
        # Botón de exportar métricas
        if st.button("📊 Exportar Métricas"):
            export_data = get_api_data("/api/metrics/export")
            if export_data:
                st.success("✅ Métricas exportadas exitosamente")
                st.json(export_data)
    
    # Verificar estado de la API
    health_data = get_api_data("/api/health")
    
    if not health_data:
        st.error("❌ No se puede conectar con la API de PromptGen Enterprise")
        st.stop()
    
    # Indicador de estado del sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        system_status = health_data.get("system_health", "unknown")
        status_color = {
            "healthy": "🟢",
            "warning": "🟡", 
            "critical": "🔴",
            "unknown": "⚪"
        }.get(system_status, "⚪")
        
        st.metric(
            label="Estado del Sistema",
            value=f"{status_color} {system_status.title()}"
        )
    
    with col2:
        uptime = health_data.get("uptime", 0)
        st.metric(
            label="Uptime",
            value=f"{uptime:.1f}%"
        )
    
    with col3:
        active_alerts = health_data.get("active_alerts", 0)
        st.metric(
            label="Alertas Activas",
            value=active_alerts,
            delta=f"Críticas: {health_data.get('critical_alerts', 0)}"
        )
    
    with col4:
        version = health_data.get("version", "Unknown")
        st.metric(
            label="Versión",
            value=version
        )
    
    st.markdown("---")
    
    # Obtener métricas del dashboard
    dashboard_data = get_api_data("/api/metrics/dashboard")
    
    if not dashboard_data:
        st.warning("⚠️ No se pudieron obtener las métricas del dashboard")
        return
    
    # Métricas de rendimiento actuales
    st.header("📊 Métricas de Rendimiento")
    
    current_perf = dashboard_data.get("current_performance", {})
    
    if current_perf:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_usage = current_perf.get("cpu_usage", 0)
            fig_cpu = create_gauge_chart(cpu_usage, "CPU Usage (%)")
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            memory_usage = current_perf.get("memory_usage", 0)
            fig_memory = create_gauge_chart(memory_usage, "Memory Usage (%)")
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            error_rate = current_perf.get("error_rate", 0)
            fig_error = create_gauge_chart(
                error_rate, 
                "Error Rate (%)", 
                max_value=20,
                color_ranges=[(0, 2, "green"), (2, 5, "yellow"), (5, 20, "red")]
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        # Métricas adicionales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Tiempo de Respuesta",
                format_metric_value(current_perf.get("response_time", 0), "time")
            )
        
        with col2:
            st.metric(
                "Sesiones Activas",
                format_metric_value(current_perf.get("active_sessions", 0))
            )
        
        with col3:
            st.metric(
                "Total Requests",
                format_metric_value(current_perf.get("total_requests", 0))
            )
        
        with col4:
            st.metric(
                "Tiempo Carga Modelo",
                format_metric_value(current_perf.get("model_load_time", 0), "time")
            )
    
    st.markdown("---")
    
    # Métricas de negocio
    st.header("💼 Métricas de Negocio")
    
    current_business = dashboard_data.get("current_business", {})
    
    if current_business:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prompts Procesados",
                format_metric_value(current_business.get("total_prompts_processed", 0))
            )
        
        with col2:
            st.metric(
                "Mejoras Exitosas",
                format_metric_value(current_business.get("successful_improvements", 0))
            )
        
        with col3:
            st.metric(
                "Tasa de Conversión",
                format_metric_value(current_business.get("conversion_rate", 0), "percentage")
            )
        
        with col4:
            st.metric(
                "Mejora Promedio",
                format_metric_value(current_business.get("average_quality_improvement", 0), "percentage")
            )
        
        # Distribución de uso de modelos
        model_usage = current_business.get("model_usage_distribution", {})
        if model_usage:
            st.subheader("🤖 Distribución de Uso de Modelos")
            
            df_models = pd.DataFrame(
                list(model_usage.items()),
                columns=["Modelo", "Uso"]
            )
            
            fig_models = px.pie(
                df_models, 
                values="Uso", 
                names="Modelo",
                title="Distribución de Uso por Modelo"
            )
            st.plotly_chart(fig_models, use_container_width=True)
    
    st.markdown("---")
    
    # Tendencias de rendimiento
    st.header("📈 Tendencias de Rendimiento")
    
    performance_trends = dashboard_data.get("performance_trends", [])
    
    if performance_trends:
        df_trends = pd.DataFrame(performance_trends)
        df_trends['timestamp'] = pd.to_datetime(df_trends['timestamp'])
        
        # Gráfico de múltiples métricas
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CPU Usage", "Memory Usage", "Response Time", "Error Rate"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['timestamp'],
                y=df_trends['cpu_usage'],
                name="CPU %",
                line=dict(color="blue")
            ),
            row=1, col=1
        )
        
        # Memory Usage
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['timestamp'],
                y=df_trends['memory_usage'],
                name="Memory %",
                line=dict(color="green")
            ),
            row=1, col=2
        )
        
        # Response Time
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['timestamp'],
                y=df_trends['response_time'],
                name="Response Time (s)",
                line=dict(color="orange")
            ),
            row=2, col=1
        )
        
        # Error Rate
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['timestamp'],
                y=df_trends['error_rate'],
                name="Error Rate %",
                line=dict(color="red")
            ),
            row=2, col=2
        )
        
        fig_trends.update_layout(
            height=600,
            title_text="Tendencias de Rendimiento (Últimas 6 horas)",
            showlegend=False
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    st.markdown("---")
    
    # Alertas recientes
    st.header("🚨 Alertas Recientes")
    
    recent_alerts = dashboard_data.get("recent_alerts", [])
    
    if recent_alerts:
        # Filtrar alertas según configuración
        if not show_all_alerts:
            recent_alerts = [a for a in recent_alerts if a.get("severity") in ["high", "critical"]]
        
        if recent_alerts:
            df_alerts = pd.DataFrame(recent_alerts)
            df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
            df_alerts = df_alerts.sort_values('timestamp', ascending=False)
            
            # Colorear según severidad
            def get_severity_color(severity):
                colors = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }
                return colors.get(severity, "⚪")
            
            df_alerts['Severidad'] = df_alerts['severity'].apply(
                lambda x: f"{get_severity_color(x)} {x.title()}"
            )
            
            # Mostrar tabla de alertas
            st.dataframe(
                df_alerts[['timestamp', 'Severidad', 'metric', 'value', 'threshold', 'message']].rename(columns={
                    'timestamp': 'Timestamp',
                    'metric': 'Métrica',
                    'value': 'Valor',
                    'threshold': 'Umbral',
                    'message': 'Mensaje'
                }),
                use_container_width=True
            )
        else:
            st.success("✅ No hay alertas críticas o altas recientes")
    else:
        st.info("ℹ️ No hay alertas recientes")
    
    st.markdown("---")
    
    # Información del sistema
    with st.expander("🔧 Información del Sistema"):
        st.json(health_data)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        try:
            st.rerun()
        except AttributeError:
            # Fallback para versiones anteriores de Streamlit
            st.experimental_rerun()

if __name__ == "__main__":
    main() 