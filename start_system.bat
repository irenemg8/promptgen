@echo off
echo =============================================
echo    🤖 PromptGen - Sistema de Chat con Documentos
echo =============================================
echo.

echo 🔍 Verificando Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama no está instalado
    echo 📥 Descarga desde: https://ollama.com/download
    echo 🔄 Ejecuta este script después de instalar Ollama
    pause
    exit /b 1
)

echo ✅ Ollama instalado
echo.

echo 🔍 Verificando modelos...
ollama list | findstr "llama3.2:3b" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 Descargando modelo llama3.2:3b...
    ollama pull llama3.2:3b
)

ollama list | findstr "mxbai-embed-large" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 Descargando modelo mxbai-embed-large...
    ollama pull mxbai-embed-large
)

echo ✅ Modelos listos
echo.

echo 🚀 Iniciando sistema...
echo.
echo 📋 Instrucciones:
echo 1. Este script iniciará el BACKEND (API)
echo 2. Abre una NUEVA terminal y ejecuta: npm run dev
echo 3. Ve a: http://localhost:3000/chat
echo.
echo 🔄 Iniciando servidor API...
echo.

python api_server.py 