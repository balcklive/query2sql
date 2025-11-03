#!/bin/bash

# Query2SQL MCP Service Startup Script
# Using streamable-http transport for streaming responses

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Query2SQL MCP Service${NC}"
echo -e "${BLUE}  Transport: streamable-http (Streamable HTTP)${NC}"
echo -e "${BLUE}  Port: 8000${NC}"
echo -e "${BLUE}  Host: 0.0.0.0 (all interfaces)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check configuration source
if [ -f .env ]; then
    echo -e "${GREEN}✓ Found .env file, will load configuration from it${NC}"
elif [ -n "$DATABASE_URL" ]; then
    echo -e "${GREEN}✓ Using DATABASE_URL from environment variables${NC}"
else
    echo -e "${GREEN}✗ Error: DATABASE_URL not configured${NC}"
    echo ""
    echo "Please configure DATABASE_URL either by:"
    echo "  1. Creating .env file with: DATABASE_URL=postgresql://..."
    echo "  2. Setting environment variable: export DATABASE_URL=postgresql://..."
    exit 1
fi

# Start the service
echo -e "${GREEN}Starting service...${NC}"
echo ""

uv run python main.py --transport=streamable-http --host=0.0.0.0 --port=8000
