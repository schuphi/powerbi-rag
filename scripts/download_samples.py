#!/usr/bin/env python3
"""
Download Adventure Works sample PBIX files for testing
"""

import os
import urllib.request
from pathlib import Path

def download_file(url: str, filename: str, target_dir: Path) -> bool:
    """Download file from URL to target directory."""
    target_path = target_dir / filename
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded: {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    """Download Adventure Works sample files."""
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Adventure Works sample PBIX files...")
    
    # Microsoft official samples
    samples = [
        {
            "name": "Adventure Works Sales Sample.pbix",
            "url": "https://github.com/microsoft/powerbi-desktop-samples/raw/main/AdventureWorks%20Sales%20Sample/AdventureWorks%20Sales.pbix",
            "description": "Adventure Works Sales analysis with customer, product, and sales data"
        },
        {
            "name": "Adventure Works DW 2020.pbix", 
            "url": "https://github.com/microsoft/powerbi-desktop-samples/raw/main/DAX/Adventure%20Works%20DW%202020.pbix",
            "description": "Data warehouse model with DAX measures for learning"
        }
    ]
    
    downloaded_files = []
    
    for sample in samples:
        success = download_file(sample["url"], sample["name"], data_dir)
        if success:
            downloaded_files.append(sample)
    
    print(f"\n downloaded {len(downloaded_files)} sample files")
    
    if downloaded_files:
        print("\nAvailable samples:")
        for sample in downloaded_files:
            print(f"- {sample['name']}")
            print(f"  {sample['description']}")
        
        print("Set up your API keys in .env file")
        print("Run: python -m powerbi_rag.cli extract-pbix './data/raw/Adventure Works Sales Sample.pbix'")
        print("Verify extraction: python scripts/simple_test.py")
        print("Start the web interface: python -m powerbi_rag.cli start-ui")
        print("Upload and analyze the PBIX files!")
    
    return len(downloaded_files) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
