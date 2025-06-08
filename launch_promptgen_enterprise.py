#!/usr/bin/env python3
"""
PromptGen Enterprise - Launcher Único
====================================

Script principal para lanzar todos los servicios de PromptGen Enterprise:
- API FastAPI (puerto 8000)
- Frontend Next.js (puerto 3000) 
- Dashboard Streamlit (puerto 8501)

Uso:
    python launch_promptgen_enterprise.py
    python launch_promptgen_enterprise.py --api-only
    python launch_promptgen_enterprise.py --frontend-only
    python launch_promptgen_enterprise.py --dashboard-only
"""

import subprocess
import sys
import os
import time
import signal
import threading
import argparse
import webbrowser
from pathlib import Path

class PromptGenLauncher:
    """Launcher empresarial para todos los servicios de PromptGen"""
    
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        self.running = True
        
    def setup_signal_handlers(self):
        """Configura manejadores de señales para cierre limpio"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Recibida señal {signum}, cerrando servicios...")
            self.shutdown_all()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def check_dependencies(self):
        """Verifica que las dependencias estén instaladas"""
        print("🔍 Verificando dependencias...")
        
        # Verificar Python
        if sys.version_info < (3, 8):
            print("❌ Se requiere Python 3.8 o superior")
            return False
            
        # Verificar Node.js para el frontend
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5, shell=True)
            if result.returncode == 0:
                print(f"✅ Node.js: {result.stdout.strip()}")
            else:
                print("⚠️ Node.js no encontrado - Frontend no disponible")
        except:
            print("⚠️ Node.js no encontrado - Frontend no disponible")
            
        # Verificar dependencias Python críticas
        critical_packages = ['fastapi', 'uvicorn', 'streamlit']
        missing_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
                
        if missing_packages:
            print(f"\n💡 Instala dependencias faltantes:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
            
        return True
    
    def start_api_server(self):
        """Inicia el servidor API FastAPI"""
        print("🚀 Iniciando API FastAPI en puerto 8000...")
        
        try:
            # Verificar que el archivo de la API existe
            api_file = self.project_root / "api_server.py"
            if not api_file.exists():
                print(f"❌ No se encuentra {api_file}")
                return None
                
            # Iniciar servidor API
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "api_server:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=self.project_root)
            
            print("✅ API FastAPI iniciada")
            return process
            
        except Exception as e:
            print(f"❌ Error iniciando API: {e}")
            return None
    
    def start_frontend(self):
        """Inicia el frontend Next.js"""
        print("🎯 Iniciando Frontend Next.js en puerto 3000...")
        
        try:
            # Verificar que package.json existe
            package_json = self.project_root / "package.json"
            if not package_json.exists():
                print("⚠️ package.json no encontrado - Frontend no disponible")
                return None
                
            # Verificar si node_modules existe o si falta cross-env
            node_modules = self.project_root / "node_modules"
            cross_env_path = node_modules / ".bin" / "cross-env.cmd"
            
            if not node_modules.exists() or not cross_env_path.exists():
                print("📦 Instalando dependencias de Node.js...")
                install_process = subprocess.run(['npm', 'install'], 
                                               cwd=self.project_root, 
                                               capture_output=True, text=True, shell=True)
                if install_process.returncode != 0:
                    print(f"❌ Error instalando dependencias: {install_process.stderr}")
                    print("💡 Intentando con npm ci...")
                    # Intentar con npm ci como alternativa
                    ci_process = subprocess.run(['npm', 'ci'], 
                                              cwd=self.project_root, 
                                              capture_output=True, text=True, shell=True)
                    if ci_process.returncode != 0:
                        print(f"❌ Error con npm ci: {ci_process.stderr}")
                        return None
                    
            # Iniciar frontend
            print("🔄 Iniciando servidor Next.js...")
            process = subprocess.Popen([
                'npm', 'run', 'dev'
            ], cwd=self.project_root, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Esperar un poco para ver si inicia correctamente
            time.sleep(3)
            if process.poll() is None:
                print("✅ Frontend Next.js iniciado")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Frontend falló al iniciar:")
                print(f"STDOUT: {stdout.decode() if stdout else 'N/A'}")
                print(f"STDERR: {stderr.decode() if stderr else 'N/A'}")
                return None
            
        except Exception as e:
            print(f"❌ Error iniciando frontend: {e}")
            return None
    
    def start_dashboard(self):
        """Inicia el dashboard Streamlit"""
        print("📊 Iniciando Dashboard Streamlit en puerto 8501...")
        
        try:
            # Verificar que el archivo del dashboard existe
            dashboard_file = self.project_root / "enterprise_dashboard.py"
            if not dashboard_file.exists():
                print(f"❌ No se encuentra {dashboard_file}")
                return None
                
            # Iniciar dashboard
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "enterprise_dashboard.py",
                "--server.port", "8501",
                "--server.headless", "true",
                "--server.runOnSave", "true"
            ], cwd=self.project_root)
            
            print("✅ Dashboard Streamlit iniciado")
            return process
            
        except Exception as e:
            print(f"❌ Error iniciando dashboard: {e}")
            return None
    
    def wait_for_services(self):
        """Espera a que los servicios estén disponibles"""
        import requests
        
        services = [
            ("API FastAPI", "http://localhost:8000/api/health"),
            ("Frontend Next.js", "http://localhost:3000"),
            ("Dashboard Streamlit", "http://localhost:8501")
        ]
        
        print("\n⏳ Esperando que los servicios estén disponibles...")
        
        for name, url in services:
            for attempt in range(15):  # 15 intentos = 30 segundos
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code < 500:
                        print(f"✅ {name} - DISPONIBLE")
                        break
                except:
                    pass
                    
                if attempt == 14:
                    print(f"⚠️ {name} - NO RESPONDE")
                else:
                    time.sleep(2)
    
    def open_browser(self):
        """Abre los servicios en el navegador"""
        print("\n🌐 Abriendo servicios en el navegador...")
        
        # Solo abrir URLs de servicios que están funcionando
        import requests
        
        urls = [
            ("Frontend", "http://localhost:3000/promptgen"),
            ("Dashboard", "http://localhost:8501"),
            ("API Docs", "http://localhost:8000/docs")
        ]
        
        for name, url in urls:
            try:
                # Verificar si el servicio responde
                response = requests.get(url.replace('/promptgen', '').replace('/docs', '/api/health'), timeout=2)
                if response.status_code < 500:
                    webbrowser.open(url)
                    print(f"✅ Abriendo {name}")
                    time.sleep(1)
                else:
                    print(f"⚠️ {name} no disponible")
            except:
                print(f"⚠️ {name} no disponible")
    
    def show_status(self):
        """Muestra el estado de los servicios"""
        print("\n" + "="*60)
        print("🎉 PromptGen Enterprise - SERVICIOS ACTIVOS")
        print("="*60)
        print("🎯 Frontend Principal: http://localhost:3000/promptgen")
        print("📊 Dashboard Empresarial: http://localhost:8501")
        print("🔍 API Health Check: http://localhost:8000/api/health")
        print("📚 Documentación API: http://localhost:8000/docs")
        print("="*60)
        print("\n💡 Funcionalidades disponibles:")
        print("   • Mejora iterativa empresarial de prompts")
        print("   • Monitoreo en tiempo real con métricas")
        print("   • Dashboard interactivo con alertas")
        print("   • API RESTful con 8 endpoints")
        print("\n🛑 Presiona Ctrl+C para detener todos los servicios")
    
    def monitor_processes(self):
        """Monitorea los procesos y los reinicia si fallan"""
        while self.running:
            for i, process in enumerate(self.processes):
                if process and process.poll() is not None:
                    print(f"⚠️ Proceso {i} terminó inesperadamente")
                    
            time.sleep(5)
    
    def shutdown_all(self):
        """Cierra todos los servicios de forma limpia"""
        print("\n🛑 Cerrando servicios...")
        self.running = False
        
        for i, process in enumerate(self.processes):
            if process:
                try:
                    print(f"🔄 Cerrando proceso {i}...")
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚡ Forzando cierre del proceso {i}...")
                    process.kill()
                except:
                    pass
                    
        print("✅ Todos los servicios cerrados")
    
    def launch_all(self):
        """Lanza todos los servicios"""
        print("🚀 PromptGen Enterprise - Launcher Único")
        print("="*50)
        
        if not self.check_dependencies():
            print("\n❌ Dependencias faltantes. Abortando...")
            return False
            
        self.setup_signal_handlers()
        
        # Iniciar servicios
        api_process = self.start_api_server()
        if api_process:
            self.processes.append(api_process)
            
        frontend_process = self.start_frontend()
        if frontend_process:
            self.processes.append(frontend_process)
            
        dashboard_process = self.start_dashboard()
        if dashboard_process:
            self.processes.append(dashboard_process)
            
        if not self.processes:
            print("❌ No se pudo iniciar ningún servicio")
            return False
            
        # Verificar que al menos la API esté funcionando
        api_running = any(p for p in self.processes if p)
        if not api_running:
            print("❌ La API es crítica y no se pudo iniciar")
            return False
            
        print(f"✅ {len([p for p in self.processes if p])} servicios iniciados correctamente")
            
        # Esperar a que estén disponibles
        self.wait_for_services()
        
        # Abrir navegador solo para servicios disponibles
        self.open_browser()
        
        # Mostrar estado
        self.show_status()
        
        # Monitorear procesos
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Mantener vivo
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
            
        return True
    
    def launch_api_only(self):
        """Lanza solo la API"""
        print("🚀 Iniciando solo API FastAPI...")
        self.setup_signal_handlers()
        
        api_process = self.start_api_server()
        if api_process:
            self.processes.append(api_process)
            print("✅ API disponible en: http://localhost:8000")
            print("📚 Documentación: http://localhost:8000/docs")
            
            try:
                api_process.wait()
            except KeyboardInterrupt:
                self.shutdown_all()
                
    def launch_frontend_only(self):
        """Lanza solo el frontend"""
        print("🎯 Iniciando solo Frontend Next.js...")
        self.setup_signal_handlers()
        
        frontend_process = self.start_frontend()
        if frontend_process:
            self.processes.append(frontend_process)
            print("✅ Frontend disponible en: http://localhost:3000/promptgen")
            
            try:
                frontend_process.wait()
            except KeyboardInterrupt:
                self.shutdown_all()
                
    def launch_dashboard_only(self):
        """Lanza solo el dashboard"""
        print("📊 Iniciando solo Dashboard Streamlit...")
        self.setup_signal_handlers()
        
        dashboard_process = self.start_dashboard()
        if dashboard_process:
            self.processes.append(dashboard_process)
            print("✅ Dashboard disponible en: http://localhost:8501")
            
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                self.shutdown_all()

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="PromptGen Enterprise - Launcher Único",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python launch_promptgen_enterprise.py              # Lanza todos los servicios
  python launch_promptgen_enterprise.py --api-only   # Solo API
  python launch_promptgen_enterprise.py --frontend-only  # Solo Frontend
  python launch_promptgen_enterprise.py --dashboard-only # Solo Dashboard
        """
    )
    
    parser.add_argument('--api-only', action='store_true', 
                       help='Lanza solo el servidor API')
    parser.add_argument('--frontend-only', action='store_true',
                       help='Lanza solo el frontend Next.js')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Lanza solo el dashboard Streamlit')
    
    args = parser.parse_args()
    
    launcher = PromptGenLauncher()
    
    try:
        if args.api_only:
            launcher.launch_api_only()
        elif args.frontend_only:
            launcher.launch_frontend_only()
        elif args.dashboard_only:
            launcher.launch_dashboard_only()
        else:
            launcher.launch_all()
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        launcher.shutdown_all()
        sys.exit(1)

if __name__ == "__main__":
    main() 