#!/bin/bash
# Quick start script for Event Tracking System

set -e

echo "📊 Event Tracking System - Quick Start"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
python3 --version

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r examples/requirements.txt

# Check PostgreSQL
echo -e "${YELLOW}Checking PostgreSQL...${NC}"
if ! command -v psql &> /dev/null; then
    echo -e "${RED}PostgreSQL not found. Install it with:${NC}"
    echo "  Ubuntu/Debian: sudo apt install postgresql"
    echo "  macOS: brew install postgresql"
    echo "  Windows: https://www.postgresql.org/download/"
    exit 1
fi

# Create database
echo -e "${YELLOW}Setting up database...${NC}"
psql postgres -c "DROP DATABASE IF EXISTS event_tracking;"
psql postgres -c "CREATE DATABASE event_tracking;"

# Initialize tables
python3 -c "
from examples.db_models import init_db
init_db()
"

echo -e "${GREEN}✓ Database setup complete${NC}"

# Instructions
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Start the services in separate terminals:"
echo ""
echo -e "${YELLOW}Terminal 1 - Event Tracking:${NC}"
echo "  cd examples && python integrated_tracker.py"
echo ""
echo -e "${YELLOW}Terminal 2 - FastAPI Backend:${NC}"
echo "  cd examples && python api_backend.py"
echo ""
echo -e "${YELLOW}Terminal 3 - Next.js Dashboard:${NC}"
echo "  cd frontend && npm install && npm run dev"
echo ""
echo -e "${GREEN}Dashboard available at:${NC} http://localhost:3000"
echo -e "${GREEN}API available at:${NC} http://localhost:8000"
echo -e "${GREEN}API Docs:${NC} http://localhost:8000/docs"
