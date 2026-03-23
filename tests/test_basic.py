#!/usr/bin/env python3
"""
Basic functionality test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_imports():
    """Test core imports work"""
    try:
        from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
        from powerbi_rag.utils.config import settings
        print("Imports OK")
    except ImportError as e:
        raise AssertionError(f"Import failed: {e}") from e

def test_config():
    """Test configuration"""
    from powerbi_rag.utils.config import settings
    
    print(f"Environment: {settings.environment}")
    print(f"LLM model: {settings.llm_model}")
    
    if not settings.openai_api_key:
        print("Warning: No OpenAI API key set")
    if not settings.anthropic_api_key:
        print("Warning: No Anthropic API key set")

def test_pbix_extraction():
    """Test PBIX extraction if sample available"""
    from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
    
    sample_path = Path("data/raw/Adventure Works Sales Sample.pbix")
    if not sample_path.exists():
        print("No sample file found - run download_samples.py first")
        return True
    
    extractor = PBIXExtractor()
    report = extractor.extract_report(sample_path)
    artifacts = extractor.extract_artifacts(report)
    
    print(f"Extracted {len(artifacts)} artifacts from {report.name}")

def main():
    print("Testing core functionality...")
    
    test_imports()
    test_config()
    test_pbix_extraction()
    
    print("Basic tests completed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
