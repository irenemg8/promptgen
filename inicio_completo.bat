@echo off
title Sistema PromptGen - Inicio Completo
color 0a

echo.
echo ===============================================
echo  🚀 SISTEMA PROMPTGEN - INICIO COMPLETO
echo ===============================================
echo.

REM Verificar si Ollama está instalado
echo 🔍 Verificando Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama no está instalado
    echo 💡 Ejecuta primero: instalar_ollama.bat
    pause
    exit /b 1
)
echo ✅ Ollama detectado

REM Verificar si Python está instalado
echo 🔍 Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python no está instalado
    echo 💡 Instala Python desde: https://python.org/downloads
    pause
    exit /b 1
)
echo ✅ Python detectado

REM Verificar si Node.js está instalado
echo 🔍 Verificando Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js no está instalado
    echo 💡 Instala Node.js desde: https://nodejs.org
    pause
    exit /b 1
)
echo ✅ Node.js detectado

REM Verificar si las dependencias están instaladas
echo 🔍 Verificando dependencias Python...
if not exist "simple_document_system.py" (
    echo ❌ Archivos del sistema no encontrados
    echo 💡 Asegúrate de estar en la carpeta correcta
    pause
    exit /b 1
)
echo ✅ Archivos del sistema encontrados

REM Instalar dependencias Python si es necesario
echo 🔧 Instalando dependencias Python...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Error instalando dependencias Python
    echo 💡 Intentando instalación manual...
    pip install fastapi uvicorn requests cryptography pillow python-magic >nul 2>&1
)
echo ✅ Dependencias Python listas

REM Instalar dependencias Node.js si es necesario
echo 🔧 Verificando dependencias Node.js...
if not exist "node_modules" (
    echo 📦 Instalando dependencias Node.js...
    npm install >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error instalando dependencias Node.js
        echo 💡 Intenta ejecutar: npm install
        pause
        exit /b 1
    )
)
echo ✅ Dependencias Node.js listas

REM Cerrar procesos previos
echo 🔄 Cerrando procesos previos...
taskkill /f /im ollama.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Iniciar Ollama
echo 🚀 Iniciando Ollama...
start /min cmd /c "ollama serve"
timeout /t 5 /nobreak >nul

REM Verificar conexión con Ollama
echo ⏳ Verificando conexión con Ollama...
timeout /t 3 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama tardó en iniciar. Esperando...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error conectando con Ollama
        echo 💡 Verifica que Ollama esté instalado correctamente
        pause
        exit /b 1
    )
)
echo ✅ Ollama funcionando

REM Verificar modelos de Ollama
echo 🔍 Verificando modelos de Ollama...
ollama list | findstr llama3.2 >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  No se encontró el modelo llama3.2
    echo 📥 Descargando modelo básico... (esto puede tardar varios minutos)
    ollama pull llama3.2:1b
    if %errorlevel% neq 0 (
        echo ❌ Error descargando modelo
        echo 💡 Intenta manualmente: ollama pull llama3.2:1b
    ) else (
        echo ✅ Modelo llama3.2:1b instalado
    )
) else (
    echo ✅ Modelo llama3.2 disponible
)

REM Iniciar backend
echo 🚀 Iniciando backend (simple_api_server.py)...
start /min cmd /c "python simple_api_server.py"
timeout /t 5 /nobreak >nul

REM Verificar conexión con backend
echo ⏳ Verificando conexión con backend...
timeout /t 3 /nobreak >nul
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Backend tardó en iniciar. Esperando...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:8000/api/health >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error conectando con backend
        echo 💡 Verifica que simple_api_server.py esté funcionando
        pause
        exit /b 1
    )
)
echo ✅ Backend funcionando

REM Iniciar frontend
echo 🚀 Iniciando frontend (npm run dev)...
start /min cmd /c "npm run dev"
timeout /t 10 /nobreak >nul

REM Verificar conexión con frontend
echo ⏳ Verificando conexión con frontend...
timeout /t 5 /nobreak >nul
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Frontend puede estar en puerto 3001. Verificando...
    timeout /t 3 /nobreak >nul
    curl -s http://localhost:3001 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error conectando con frontend
        echo 💡 Verifica que npm run dev esté funcionando
        set frontend_port=ERROR
    ) else (
        echo ✅ Frontend funcionando en puerto 3001
        set frontend_port=3001
    )
) else (
    echo ✅ Frontend funcionando en puerto 3000
    set frontend_port=3000
)

REM Abrir navegador automáticamente
if not "%frontend_port%"=="ERROR" (
    echo 🌐 Abriendo navegador...
    timeout /t 2 /nobreak >nul
    start http://localhost:%frontend_port%/chat
)

echo.
echo ===============================================
echo  🎉 SISTEMA INICIADO EXITOSAMENTE!
echo ===============================================
echo.
echo ✅ Ollama: http://localhost:11434
echo ✅ Backend: http://localhost:8000
if not "%frontend_port%"=="ERROR" (
    echo ✅ Frontend: http://localhost:%frontend_port%
    echo.
    echo 🚀 CHAT DISPONIBLE EN:
    echo    👉 http://localhost:%frontend_port%/chat
) else (
    echo ❌ Frontend: Error de conexión
)
echo.
echo 📝 Para detener el sistema: detener_sistema.bat
echo 💡 Para reiniciar: ejecuta este archivo nuevamente
echo.
echo Los servicios están ejecutándose en segundo plano...
pause 