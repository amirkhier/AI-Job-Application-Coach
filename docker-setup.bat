@echo off
REM Docker setup script for AI Job Application Coach (Windows)
REM This script provides an easy way to get started with Docker on Windows

setlocal EnableDelayedExpansion

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║         AI Job Application Coach - Docker Setup         ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Check if Docker is installed and running
echo [INFO] Checking Docker installation...
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker info >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)
echo [SUCCESS] Docker is installed and running

REM Check Docker Compose
echo [INFO] Checking Docker Compose...
docker compose version >nul 2>&1
if !errorlevel! neq 0 (
    docker-compose --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Docker Compose is not available
        pause
        exit /b 1
    )
    set COMPOSE_CMD=docker-compose
) else (
    set COMPOSE_CMD=docker compose
)
echo [SUCCESS] Docker Compose is available

REM Setup environment file
echo [INFO] Setting up environment configuration...
if not exist .env (
    if exist .env.docker (
        copy .env.docker .env >nul
        echo [WARNING] Created .env from .env.docker template
        echo [WARNING] Please edit .env and add your OpenAI API key!
    ) else (
        echo [ERROR] .env.docker template not found
        pause
        exit /b 1
    )
) else (
    echo [INFO] .env file already exists
)

REM Check if OpenAI API key is set
findstr "your_openai_api_key_here" .env >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo [WARNING] Please update .env with your actual OpenAI API key
    echo Edit the .env file and replace 'your_openai_api_key_here' with your actual API key
    echo You can get an API key from: https://platform.openai.com/api-keys
    echo.
    pause
)

REM Build and start services
echo [INFO] Building and starting AI Job Application Coach services...

REM Start core services first
echo [INFO] Starting database services...
%COMPOSE_CMD% up --build -d mysql redis chromadb
if !errorlevel! neq 0 (
    echo [ERROR] Failed to start database services
    pause
    exit /b 1
)

echo [INFO] Waiting for databases to be ready...
timeout /t 15 /nobreak >nul

REM Start the main application
echo [INFO] Starting main application...
%COMPOSE_CMD% up --build -d app
if !errorlevel! neq 0 (
    echo [ERROR] Failed to start main application
    pause
    exit /b 1
)

echo [SUCCESS] Services are starting up...
echo [INFO] Waiting for application to be ready...

REM Wait for application to be healthy (simplified check for Windows)
set /a counter=0
:wait_loop
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if !errorlevel! equ 0 goto app_ready
set /a counter+=1
if !counter! lss 30 goto wait_loop

echo [ERROR] Application failed to start within 60 seconds
echo [INFO] Check logs with: %COMPOSE_CMD% logs app
pause
exit /b 1

:app_ready
echo [SUCCESS] AI Job Application Coach is now running!

REM Show service status
echo.
echo [INFO] Service Status:
%COMPOSE_CMD% ps

echo.
echo Available Services:
echo 🚀 Main Application: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 Health Check: http://localhost:8000/health
echo 🗄️  MySQL Database: localhost:3307 (external port)
echo 📊 ChromaDB: http://localhost:8001
echo 💾 Redis: localhost:6380 (external port)

echo.
echo Useful Commands:
echo View logs: %COMPOSE_CMD% logs -f app
echo Stop services: %COMPOSE_CMD% down
echo Restart: %COMPOSE_CMD% restart app
echo Shell access: %COMPOSE_CMD% exec app bash

echo.
echo 🎉 Setup complete!
echo Visit http://localhost:8000/docs to explore the API
echo.
pause