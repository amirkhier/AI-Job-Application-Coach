#!/bin/bash

# Docker setup script for AI Job Application Coach
# This script provides an easy way to get started with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Setup environment file
setup_environment() {
    if [ ! -f .env ]; then
        if [ -f .env.docker ]; then
            cp .env.docker .env
            print_warning "Created .env from .env.docker template"
            print_warning "Please edit .env and add your OpenAI API key!"
        else
            print_error ".env.docker template not found"
            exit 1
        fi
    else
        print_status ".env file already exists"
    fi
    
    # Check if OpenAI API key is set
    if grep -q "your_openai_api_key_here" .env; then
        print_warning "Please update .env with your actual OpenAI API key before starting the services"
        echo -e "\nEdit the .env file and replace 'your_openai_api_key_here' with your actual API key"
        echo -e "You can get an API key from: https://platform.openai.com/api-keys\n"
        read -p "Press Enter to continue after updating .env, or Ctrl+C to exit..."
    fi
}

# Build and start services
start_services() {
    print_status "Building and starting AI Job Application Coach services..."
    
    # Start core services first
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d mysql redis chromadb
    else
        docker compose up --build -d mysql redis chromadb
    fi
    
    print_status "Waiting for databases to be ready..."
    sleep 10
    
    # Start the main application
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d app
    else
        docker compose up --build -d app
    fi
    
    print_success "Services are starting up..."
    print_status "Waiting for application to be ready..."
    
    # Wait for application to be healthy
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Application failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    print_success "AI Job Application Coach is now running!"
}

# Show service status
show_status() {
    print_status "Service Status:"
    if command -v docker-compose &> /dev/null; then
        docker-compose ps
    else
        docker compose ps
    fi
    
    echo -e "\n${GREEN}Available Services:${NC}"
    echo "🚀 Main Application: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/health"
    echo "🗄️  MySQL Database: localhost:3306"
    echo "📊 ChromaDB: http://localhost:8001"
    echo "💾 Redis: localhost:6379"
    
    echo -e "\n${YELLOW}Useful Commands:${NC}"
    echo "View logs: docker-compose logs -f app"
    echo "Stop services: docker-compose down"
    echo "Restart: docker-compose restart app"
    echo "Shell access: docker-compose exec app bash"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║            AI Job Application Coach - Docker Setup       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_docker
    check_docker_compose
    setup_environment
    start_services
    show_status
    
    echo -e "\n${GREEN}🎉 Setup complete!${NC}"
    echo -e "Visit ${BLUE}http://localhost:8000/docs${NC} to explore the API"
}

# Handle script arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "stop")
        print_status "Stopping AI Job Application Coach services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting AI Job Application Coach..."
        if command -v docker-compose &> /dev/null; then
            docker-compose restart
        else
            docker compose restart
        fi
        print_success "Services restarted"
        ;;
    "logs")
        if command -v docker-compose &> /dev/null; then
            docker-compose logs -f app
        else
            docker compose logs -f app
        fi
        ;;
    "status")
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Build and start all services (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show application logs"
        echo "  status  - Show service status"
        exit 1
        ;;
esac