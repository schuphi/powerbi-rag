#!/usr/bin/env python3
"""
Generate simple demo data for portfolio documentation
"""

import json
from datetime import datetime
from pathlib import Path

def generate_demo_data():
    """Generate basic demo data showing system capabilities."""
    return {
        "system_info": {
            "name": "Power BI RAG Assistant",
            "version": "0.1.0",
            "tech_stack": ["Python", "FastAPI", "ChromaDB", "OpenAI", "Anthropic", "Gradio"]
        },
        "sample_extraction": {
            "source_file": "Adventure Works Sales Sample.pbix",
            "artifacts_generated": 2,
            "sample_artifacts": [
                {"type": "report", "name": "Adventure Works Sales Sample"},
                {"type": "dataset", "name": "Adventure Works Dataset"}
            ]
        },
        "api_endpoints": [
            {"method": "POST", "path": "/ask", "description": "Ask questions about reports"},
            {"method": "GET", "path": "/health", "description": "System health check"},
            {"method": "POST", "path": "/upload", "description": "Upload PBIX files"}
        ],
        "generated_at": datetime.now().isoformat()
    }

def main():
    """Generate portfolio assets."""
    print("Generating demo data...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate and save demo data
    demo_data = generate_demo_data()
    with open(docs_dir / "demo_data.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"Generated: {docs_dir / 'demo_data.json'}")

if __name__ == "__main__":
    main()