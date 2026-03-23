#!/usr/bin/env python3
"""
Test the Power BI RAG system with Adventure Works sample data
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
from powerbi_rag.retrieval.vector_store import ChromaVectorStore
from powerbi_rag.retrieval.rag_pipeline import PowerBIRAGPipeline
from powerbi_rag.utils.config import settings


def check_prerequisites():
    """Check if prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check API keys
    if not settings.openai_api_key or not settings.anthropic_api_key:
        print("Missing API keys - set OPENAI_API_KEY and ANTHROPIC_API_KEY")
        return False
    
    # Check sample files
    sample_files = [
        "data/raw/Adventure Works Sales Sample.pbix",
        "data/raw/Adventure Works DW 2020.pbix"
    ]
    
    available = [f for f in sample_files if Path(f).exists()]
    if not available:
        print("No sample files - run: python scripts/download_samples.py")
        return False
    
    print(f"Found {len(available)} sample file(s)")
    return True


async def test_extraction(pbix_path: Path):
    """Test PBIX extraction."""
    print(f"\nTesting extraction: {pbix_path.name}")
    
    try:
        extractor = PBIXExtractor()
        report = extractor.extract_report(pbix_path)
        artifacts = extractor.extract_artifacts(report)
        
        print(f"Extracted {len(artifacts)} artifacts")
        return artifacts
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


async def test_rag_system(artifacts):
    """Test complete RAG system."""
    print(f"\nTesting RAG system with {len(artifacts)} artifacts...")
    
    try:
        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory="./test_chroma_db",
            collection_name="adventure_works_test"
        )
        vector_store.reset_collection()
        vector_store.add_artifacts(artifacts)
        
        # Initialize RAG pipeline
        rag_pipeline = PowerBIRAGPipeline(vector_store=vector_store)
        
        # Test questions
        test_questions = [
            "What tables are in the Adventure Works report?",
            "How is Total Sales calculated?",
            "What visualizations are available?"
        ]
        
        print("   Testing questions:")
        for i, question in enumerate(test_questions, 1):
            response = await rag_pipeline.answer_question(question)
            print(f"   Q{i}: {question}")
            print(f"   A{i}: {response['answer'][:100]}...")
            print(f"       Confidence: {response['confidence']:.2f}")
        
        print("RAG system test completed")
        return True
        
    except Exception as e:
        print(f"RAG test failed: {e}")
        return False


async def cleanup():
    """Clean up test data."""
    print("\nCleaning up...")
    try:
        import shutil
        test_db = Path("./test_chroma_db")
        if test_db.exists():
            shutil.rmtree(test_db)
            print("Cleanup completed")
    except Exception as e:
        print(f"Cleanup warning: {e}")


async def main():
    """Main test function."""
    print("Adventure Works Power BI RAG Test")
    print("=" * 40)
    
    if not check_prerequisites():
        return False
    
    # Find sample file
    sample_files = [
        Path("data/raw/Adventure Works Sales Sample.pbix"),
        Path("data/raw/Adventure Works DW 2020.pbix")
    ]
    
    sample_file = next((f for f in sample_files if f.exists()), None)
    if not sample_file:
        print("No sample files available")
        return False
    
    try:
        # Test extraction
        artifacts = await test_extraction(sample_file)
        if not artifacts:
            return False
        
        # Test RAG system
        success = await test_rag_system(artifacts)
        
        if success:
            print("\nAll tests completed successfully.")
            print("\nNext steps:")
            print("1. Start UI: python -m powerbi_rag.cli start-ui")
            print("2. Upload PBIX files and start asking questions!")
        
        return success
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False
    finally:
        await cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
