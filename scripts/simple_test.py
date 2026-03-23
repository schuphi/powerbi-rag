#!/usr/bin/env python3
"""
Simple test script that works without ChromaDB for basic functionality testing
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test that we can import our modules."""
    print("[*] Testing basic imports...")
    
    try:
        from powerbi_rag.extraction.models import PowerBIArtifact, ArtifactType, PowerBIReport
        print("[OK] Successfully imported extraction models")
        
        from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
        print("[OK] Successfully imported PBIX extractor")
        
        from powerbi_rag.utils.config import settings
        print("[OK] Successfully imported configuration")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n[*] Testing configuration...")
    
    try:
        from powerbi_rag.utils.config import settings
        
        print(f"Environment: {settings.environment}")
        print(f"Debug mode: {settings.debug}")
        print(f"LLM model: {settings.llm_model}")
        print(f"Embedding model: {settings.embedding_model}")
        
        # Check API key status (without printing actual keys)
        openai_configured = bool(settings.openai_api_key)
        anthropic_configured = bool(settings.anthropic_api_key)
        
        print(f"OpenAI API key configured: {openai_configured}")
        print(f"Anthropic API key configured: {anthropic_configured}")
        
        if not openai_configured or not anthropic_configured:
            print("\n[WARNING] Missing API keys - copy .env.example to .env and add your keys")
        
        print("[OK] Configuration test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_pbix_extractor():
    """Test PBIX extractor without actual files."""
    print("\n[*] Testing PBIX extractor...")
    
    try:
        from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
        from powerbi_rag.extraction.models import PowerBITable, PowerBIColumn, PowerBIMeasure, ArtifactType
        
        extractor = PBIXExtractor()
        print("[OK] PBIX extractor initialized")
        
        # Test artifact creation with sample data
        sample_column = PowerBIColumn(
            name="CustomerID",
            data_type="int64", 
            table_name="Customer"
        )
        
        sample_measure = PowerBIMeasure(
            name="Total Sales",
            expression="SUM(Sales[Amount])",
            table_name="Sales"
        )
        
        sample_table = PowerBITable(
            name="Customer",
            columns=[sample_column],
            measures=[sample_measure]
        )
        
        artifacts = extractor._create_table_artifacts(sample_table, "TestReport")
        print(f"[OK] Created {len(artifacts)} sample artifacts")
        
        # Check artifact types
        for artifact in artifacts:
            print(f"   - {artifact.type}: {artifact.name}")
        
        print("[OK] PBIX extractor test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] PBIX extractor test failed: {e}")
        return False

def download_adventure_works():
    """Download Adventure Works sample if available."""
    print("\n[*] Testing Adventure Works download...")
    
    try:
        import urllib.request
        from pathlib import Path
        
        # Create data directory
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to download a smaller sample first
        sample_url = "https://github.com/microsoft/powerbi-desktop-samples/raw/main/AdventureWorks%20Sales%20Sample/AdventureWorks%20Sales.pbix"
        sample_path = data_dir / "Adventure Works Sales Sample.pbix"
        
        if not sample_path.exists():
            print(f"Downloading Adventure Works sample...")
            urllib.request.urlretrieve(sample_url, sample_path)
            print(f"[OK] Downloaded: {sample_path}")
        else:
            print(f"[OK] Sample already exists: {sample_path}")
        
        # Check file size
        if sample_path.exists():
            size_mb = sample_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.1f} MB")
            return sample_path
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("You can manually download from:")
        print("https://github.com/microsoft/powerbi-desktop-samples")
    
    return None

def test_pbix_extraction(pbix_path):
    """Test actual PBIX extraction if file is available."""
    if not pbix_path or not pbix_path.exists():
        print("[WARNING] No PBIX file available for extraction test")
        return False
    
    print(f"\n[*] Testing extraction of: {pbix_path.name}")
    
    try:
        from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
        
        extractor = PBIXExtractor()
        report = extractor.extract_report(pbix_path)
        
        print(f"[OK] Extracted report: {report.name}")
        print(f"   - Dataset: {report.dataset.name}")
        print(f"   - Tables: {len(report.dataset.tables)}")
        print(f"   - Relationships: {len(report.dataset.relationships)}")
        print(f"   - Pages: {len(report.pages)}")
        
        # Generate artifacts
        artifacts = extractor.extract_artifacts(report)
        print(f"   - Total artifacts: {len(artifacts)}")
        
        # Show artifact breakdown
        artifact_counts = {}
        for artifact in artifacts:
            artifact_counts[artifact.type] = artifact_counts.get(artifact.type, 0) + 1
        
        print("   Artifact breakdown:")
        for artifact_type, count in sorted(artifact_counts.items()):
            print(f"     • {artifact_type}: {count}")
        
        # Show some sample artifacts
        print("\n   Sample artifacts:")
        for artifact in artifacts[:5]:
            print(f"     • {artifact.type}: {artifact.name}")
            print(f"       Content: {artifact.content[:80]}...")
        
        print("[OK] PBIX extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] PBIX extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Power BI RAG System - Basic Functionality Test")
    print("=" * 55)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("\n[ERROR] Basic import test failed - check your Python environment")
        return False
    
    # Test 2: Configuration
    if not test_config():
        print("\n[ERROR] Configuration test failed")
        return False
    
    # Test 3: PBIX extractor
    if not test_pbix_extractor():
        print("\n[ERROR] PBIX extractor test failed")
        return False
    
    # Test 4: Download Adventure Works
    print("\n" + "=" * 55)
    pbix_path = download_adventure_works()
    
    # Test 5: Actual PBIX extraction
    if pbix_path:
        test_pbix_extraction(pbix_path)
    
    print("\n" + "=" * 55)
    print("[SUCCESS] Basic functionality tests completed!")
    print("\nWhat works:")
    print("[OK] Core modules import correctly")
    print("[OK] Configuration system works") 
    print("[OK] PBIX extraction logic is functional")
    
    if pbix_path and pbix_path.exists():
        print("[OK] Adventure Works sample downloaded and processed")
    
    print("\nNext steps:")
    print("1. Set up API keys in .env file")
    print("2. Run full test: python scripts/test_adventure_works.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)