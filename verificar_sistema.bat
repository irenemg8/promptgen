@echo off
title Verificación del Sistema PromptGen
color 0b

echo.
echo ===============================================
echo  🔍 VERIFICACIÓN DEL SISTEMA PROMPTGEN
echo ===============================================
echo.

echo 🔍 Verificando componentes del sistema...
echo.

REM Verificar Ollama
echo [1/4] Verificando Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama: NO FUNCIONA
    echo 💡 Ejecuta: ollama serve
    set ollama_ok=false
) else (
    echo ✅ Ollama: FUNCIONANDO
    set ollama_ok=true
)

REM Verificar Backend
echo [2/4] Verificando Backend...
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Backend: NO FUNCIONA
    echo 💡 Ejecuta: python simple_api_server.py
    set backend_ok=false
) else (
    echo ✅ Backend: FUNCIONANDO
    set backend_ok=true
)

REM Verificar Frontend
echo [3/4] Verificando Frontend...
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% neq 0 (
    curl -s http://localhost:3001 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Frontend: NO FUNCIONA
        echo 💡 Ejecuta: npm run dev
        set frontend_ok=false
        set frontend_port=NONE
    ) else (
        echo ✅ Frontend: FUNCIONANDO (puerto 3001)
        set frontend_ok=true
        set frontend_port=3001
    )
) else (
    echo ✅ Frontend: FUNCIONANDO (puerto 3000)
    set frontend_ok=true
    set frontend_port=3000
)

REM Verificar modelos de Ollama
echo [4/4] Verificando modelos de Ollama...
if "%ollama_ok%"=="true" (
    ollama list | findstr llama3.2 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ⚠️  Modelos: NO ENCONTRADOS
        echo 💡 Ejecuta: ollama pull llama3.2:1b
        set models_ok=false
    ) else (
        echo ✅ Modelos: DISPONIBLES
        set models_ok=true
    )
) else (
    echo ❌ Modelos: NO SE PUEDEN VERIFICAR
    set models_ok=false
)

echo.
echo ===============================================
echo  📊 RESUMEN DE VERIFICACIÓN
echo ===============================================
echo.

if "%ollama_ok%"=="true" (
    echo ✅ Ollama: http://localhost:11434
) else (
    echo ❌ Ollama: NO DISPONIBLE
)

if "%backend_ok%"=="true" (
    echo ✅ Backend: http://localhost:8000
) else (
    echo ❌ Backend: NO DISPONIBLE
)

if "%frontend_ok%"=="true" (
    echo ✅ Frontend: http://localhost:%frontend_port%
) else (
    echo ❌ Frontend: NO DISPONIBLE
)

if "%models_ok%"=="true" (
    echo ✅ Modelos: LISTOS
) else (
    echo ❌ Modelos: NO DISPONIBLES
)

echo.

REM Determinar estado general
if "%ollama_ok%"=="true" if "%backend_ok%"=="true" if "%frontend_ok%"=="true" if "%models_ok%"=="true" (
    echo 🎉 SISTEMA COMPLETAMENTE FUNCIONAL!
    echo.
    echo 🚀 CHAT DISPONIBLE EN:
    echo    👉 http://localhost:%frontend_port%/chat
    echo.
    echo ¿Abrir en navegador? (s/n)
    set /p abrir_navegador=
    if /i "%abrir_navegador%"=="s" (
        start http://localhost:%frontend_port%/chat
    )
) else (
    echo ⚠️  SISTEMA CON PROBLEMAS
    echo.
    echo 💡 SOLUCIONES RÁPIDAS:
    echo.
    if "%ollama_ok%"=="false" (
        echo 🔧 Para Ollama: instalar_ollama.bat
    )
    if "%backend_ok%"=="false" (
        echo 🔧 Para Backend: python simple_api_server.py
    )
    if "%frontend_ok%"=="false" (
        echo 🔧 Para Frontend: npm run dev
    )
    if "%models_ok%"=="false" (
        echo 🔧 Para Modelos: ollama pull llama3.2:1b
    )
    echo.
    echo 🚀 O ejecuta: inicio_completo.bat
)

echo.
pause 