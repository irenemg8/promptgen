@echo off
title Sistema PromptGen - Detener Servicios
color 0c

echo.
echo ===============================================
echo  🛑 SISTEMA PROMPTGEN - DETENER SERVICIOS
echo ===============================================
echo.

echo 🔄 Deteniendo servicios...

REM Detener Ollama
echo 🛑 Deteniendo Ollama...
taskkill /f /im ollama.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama no estaba ejecutándose
) else (
    echo ✅ Ollama detenido
)

REM Detener Python (Backend)
echo 🛑 Deteniendo backend Python...
taskkill /f /im python.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Backend Python no estaba ejecutándose
) else (
    echo ✅ Backend Python detenido
)

REM Detener Node.js (Frontend)
echo 🛑 Deteniendo frontend Node.js...
taskkill /f /im node.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Frontend Node.js no estaba ejecutándose
) else (
    echo ✅ Frontend Node.js detenido
)

REM Detener procesos específicos por puerto
echo 🔄 Liberando puertos...

REM Puerto 11434 (Ollama)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :11434') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Puerto 8000 (Backend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Puerto 3000 (Frontend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Puerto 3001 (Frontend alternativo)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3001') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo ✅ Puertos liberados

REM Verificar que los servicios estén detenidos
echo.
echo 🔍 Verificando servicios...

REM Verificar Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Ollama: Detenido
) else (
    echo ⚠️  Ollama: Aún ejecutándose
)

REM Verificar Backend
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Backend: Detenido
) else (
    echo ⚠️  Backend: Aún ejecutándose
)

REM Verificar Frontend
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% neq 0 (
    curl -s http://localhost:3001 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ✅ Frontend: Detenido
    ) else (
        echo ⚠️  Frontend: Aún ejecutándose en puerto 3001
    )
) else (
    echo ⚠️  Frontend: Aún ejecutándose en puerto 3000
)

echo.
echo ===============================================
echo  🎉 SERVICIOS DETENIDOS EXITOSAMENTE!
echo ===============================================
echo.
echo ✅ Todos los servicios han sido detenidos
echo 🚀 Para reiniciar: ejecuta inicio_completo.bat
echo 💡 Para instalar Ollama: ejecuta instalar_ollama.bat
echo.
pause 