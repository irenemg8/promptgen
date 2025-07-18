@echo off
echo.
echo ===============================================
echo  INSTALADOR AUTOMATICO DE OLLAMA PARA WINDOWS
echo ===============================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python no está instalado. Instalando...
    echo 📥 Descargando Python...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe' -OutFile 'python-installer.exe'"
    echo 🚀 Ejecutando instalador de Python...
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-installer.exe
    echo ✅ Python instalado
) else (
    echo ✅ Python ya está instalado
)

REM Verificar si Ollama está instalado
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo 📦 Ollama no está instalado. Instalando...
    echo 📥 Descargando Ollama...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'"
    echo 🚀 Ejecutando instalador de Ollama...
    echo ⚠️  IMPORTANTE: Acepta todas las opciones por defecto
    start /wait OllamaSetup.exe
    del OllamaSetup.exe
    echo ✅ Ollama instalado
) else (
    echo ✅ Ollama ya está instalado
)

REM Iniciar servicio de Ollama
echo.
echo 🚀 Iniciando servicio de Ollama...
start /B ollama serve
timeout /t 5 /nobreak >nul

REM Verificar si el servicio está corriendo
echo ⏳ Verificando conexión...
timeout /t 3 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  El servicio tardó en iniciar. Esperando...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error conectando con Ollama
        echo 💡 Intenta ejecutar manualmente: ollama serve
        pause
        exit /b 1
    )
)

echo ✅ Ollama está funcionando correctamente

REM Verificar modelos instalados
echo.
echo 🔍 Verificando modelos instalados...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  No se pudieron verificar los modelos
) else (
    echo 📚 Modelos instalados:
    ollama list
)

REM Preguntar si instalar modelo básico
echo.
set /p instalar_modelo="¿Quieres instalar el modelo básico llama3.2:1b (1.3GB)? (s/n): "
if /i "%instalar_modelo%"=="s" (
    echo 📥 Descargando modelo llama3.2:1b... (esto puede tardar varios minutos)
    ollama pull llama3.2:1b
    if %errorlevel% neq 0 (
        echo ❌ Error descargando modelo
    ) else (
        echo ✅ Modelo llama3.2:1b instalado exitosamente
    )
) else (
    echo ⏭️  Saltando instalación de modelo
)

REM Probar conexión
echo.
echo 🧪 Probando conexión final...
curl -s -X POST http://localhost:11434/api/generate -d "{\"model\":\"llama3.2:1b\",\"prompt\":\"Hello\",\"stream\":false}" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  No se pudo probar el modelo, pero Ollama está funcionando
) else (
    echo ✅ Prueba exitosa
)

REM Crear script de inicio
echo.
echo 📝 Creando script de inicio...
echo @echo off > start_ollama.bat
echo echo Iniciando Ollama... >> start_ollama.bat
echo ollama serve >> start_ollama.bat
echo pause >> start_ollama.bat
echo ✅ Script creado: start_ollama.bat

echo.
echo ===============================================
echo  🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE!
echo ===============================================
echo.
echo ✅ Ollama está instalado y funcionando
echo 🌐 Puedes usar el chat ahora
echo 📝 Para reiniciar Ollama: ejecuta start_ollama.bat
echo.
echo 🚀 Para usar el sistema completo:
echo    1. Ejecuta: python simple_api_server.py
echo    2. Ejecuta: npm run dev
echo    3. Visita: http://localhost:3001/chat
echo.
pause 