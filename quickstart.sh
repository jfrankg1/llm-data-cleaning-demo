#!/bin/bash
# LLM Data Cleaning System - Quick Start Script

echo "🚀 LLM Data Cleaning System - Quick Start"
echo "========================================="

# Check Python
echo "✓ Checking Python installation..."
python3 --version

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f .env ]; then
    echo "✓ Creating .env from example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your database credentials"
fi

# Instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Update .env with your database credentials"
echo "2. Run: streamlit run streamlit_app.py"
echo ""
echo "For Streamlit Cloud deployment, see DEPLOYMENT_GUIDE.md"