#!/usr/bin/env python3
"""
Setup script for Power BI RAG Assistant
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main setup function."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("Setting up Power BI RAG Assistant...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("Copying .env.example to .env")
        if Path(".env.example").exists():
            subprocess.run("cp .env.example .env", shell=True)
            print("Please update .env with your API keys.")
        else:
            print(".env.example not found.")
            return False
    
    # Install dependencies
    print("Installing dependencies...")
    if not run_command("pip install -e .[dev]"):
        print("Failed to install dependencies")
        return False
    
    # Setup pre-commit hooks
    print("Setting up pre-commit hooks...")
    if not run_command("pre-commit install"):
        print("Failed to setup pre-commit hooks")
        return False
    
    # Create data directories
    print("Creating data directories...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    
    print("Setup complete.")
    print("\nNext steps:")
    print("1. Update .env with your API keys")
    print("2. Add sample PBIX files to data/raw/")
    print("3. Run: python -m powerbi_rag.cli extract-pbix")
    print("4. Run: python -m powerbi_rag.cli start-api")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
