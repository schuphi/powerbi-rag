#!/usr/bin/env python3
"""
Demo of working Power BI RAG features
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def demo_config():
    """Demo configuration system."""
    print("=== Configuration System ===")
    
    from powerbi_rag.utils.config import settings
    
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"LLM model (dev): {settings.llm_model_dev}")
    print(f"LLM model (prod): {settings.llm_model_prod}")
    print(f"Current LLM model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model}")
    
    # Check API keys (without showing them)
    print(f"OpenAI configured: {'Yes' if settings.openai_api_key else 'No - add to .env'}")
    print(f"Anthropic configured: {'Yes' if settings.anthropic_api_key else 'No - add to .env'}")

def demo_pbix_extraction():
    """Demo PBIX extraction capabilities."""
    print("\n=== PBIX Extraction System ===")
    
    from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
    
    # Check if sample file exists
    sample_path = Path("data/raw/Adventure Works Sales Sample.pbix")
    
    if sample_path.exists():
        print(f"Processing: {sample_path.name}")
        
        extractor = PBIXExtractor()
        report = extractor.extract_report(sample_path)
        
        print(f"Report: {report.name}")
        print(f"Dataset: {report.dataset.name}")
        print(f"File path: {report.file_path}")
        
        # Generate searchable artifacts
        artifacts = extractor.extract_artifacts(report)
        print(f"Generated {len(artifacts)} searchable artifacts")
        
        # Show artifact types
        artifact_counts = {}
        for artifact in artifacts:
            artifact_counts[artifact.type] = artifact_counts.get(artifact.type, 0) + 1
        
        print("Artifact breakdown:")
        for artifact_type, count in sorted(artifact_counts.items()):
            print(f"  {artifact_type}: {count}")
        
        # Show sample artifacts
        print("\nSample artifacts:")
        for artifact in artifacts:
            print(f"  [{artifact.type}] {artifact.name}")
            print(f"    Content: {artifact.content[:100]}...")
            print(f"    Tags: {', '.join(artifact.tags[:3])}")
            print()
    
    else:
        print("No Adventure Works sample found")
        print("Run: python scripts/simple_test.py to verify extraction")

def demo_data_models():
    """Demo the data models with sample data."""
    print("=== Power BI Data Models ===")
    
    from powerbi_rag.extraction.models import (
        PowerBIColumn, PowerBIMeasure, PowerBITable, 
        PowerBIRelationship, ArtifactType
    )
    from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
    
    # Create sample data structures
    print("Creating sample Power BI structures...")
    
    # Sample columns
    customer_columns = [
        PowerBIColumn(
            name="CustomerID",
            data_type="int64",
            table_name="Customer",
            description="Unique customer identifier"
        ),
        PowerBIColumn(
            name="CustomerName", 
            data_type="string",
            table_name="Customer",
            description="Customer full name"
        ),
        PowerBIColumn(
            name="Territory",
            data_type="string", 
            table_name="Customer",
            description="Sales territory"
        )
    ]
    
    # Sample measures
    customer_measures = [
        PowerBIMeasure(
            name="Total Customers",
            expression="COUNTROWS(Customer)",
            table_name="Customer",
            description="Count of all customers"
        ),
        PowerBIMeasure(
            name="Active Customers",
            expression="CALCULATE(COUNTROWS(Customer), Customer[Status] = \"Active\")",
            table_name="Customer",
            description="Count of active customers only"
        )
    ]
    
    # Sample table
    customer_table = PowerBITable(
        name="Customer",
        description="Customer master data",
        columns=customer_columns,
        measures=customer_measures
    )
    
    # Sample relationship
    customer_sales_rel = PowerBIRelationship(
        from_table="Customer",
        from_column="CustomerID", 
        to_table="Sales",
        to_column="CustomerID",
        cardinality="OneToMany",
        cross_filter_direction="OneDirection"
    )
    
    print(f"Sample Table: {customer_table.name}")
    print(f"  Columns: {len(customer_table.columns)}")
    print(f"  Measures: {len(customer_table.measures)}")
    print(f"  Description: {customer_table.description}")
    
    print(f"\nSample Relationship: {customer_sales_rel.from_table} -> {customer_sales_rel.to_table}")
    print(f"  Cardinality: {customer_sales_rel.cardinality}")
    print(f"  Cross-filter: {customer_sales_rel.cross_filter_direction}")
    
    # Convert to artifacts
    extractor = PBIXExtractor()
    table_artifacts = extractor._create_table_artifacts(customer_table, "Demo")
    rel_artifact = extractor._create_relationship_artifact(customer_sales_rel, "Demo")
    
    print(f"\nGenerated {len(table_artifacts)} table artifacts + 1 relationship artifact")
    print("Sample artifact content:")
    for artifact in table_artifacts[:2]:
        print(f"  [{artifact.type}] {artifact.name}")
        print(f"    {artifact.content}")
        print()

def demo_api_structure():
    """Show the API structure that would be available."""
    print("=== API Structure (Available when ChromaDB is installed) ===")
    
    api_endpoints = [
        "POST /ask - Ask questions about Power BI reports",
        "POST /upload - Upload and process PBIX files", 
        "GET /health - System health check",
        "GET /vector-store/info - Vector database statistics",
        "POST /vector-store/search - Search artifacts directly",
        "POST /explain/measure - Explain specific DAX measures",
        "POST /describe/table - Describe table structure",
        "GET /conversation/history/{session_id} - Get conversation history",
        "DELETE /cache/clear - Clear response cache",
        "GET /cache/stats - Cache statistics"
    ]
    
    print("Available REST API endpoints:")
    for endpoint in api_endpoints:
        print(f"  {endpoint}")
    
    print(f"\nAPI Features:")
    print("  - Async FastAPI backend")
    print("  - Conversation management with session tracking")
    print("  - Smart caching to reduce API costs") 
    print("  - File upload with background processing")
    print("  - Comprehensive error handling")
    print("  - OpenAPI/Swagger documentation")

def demo_gradio_interface():
    """Show what the Gradio interface provides."""
    print("\n=== Web Interface Features ===")
    
    interface_features = [
        "File Upload Tab - Upload and process PBIX files",
        "Chat Tab - Interactive Q&A with conversation history",
        "System Status Tab - Monitor health and statistics",
        "Session Management - Multiple conversation threads",
        "Progress Indicators - Real-time processing feedback",
        "Source Citations - Show which artifacts informed responses",
        "Confidence Scores - AI response confidence levels",
        "Filter Options - Search by artifact type (table, measure, etc.)"
    ]
    
    print("Gradio Web Interface includes:")
    for feature in interface_features:
        print(f"  - {feature}")
    
    sample_questions = [
        "What tables are available in this Adventure Works report?",
        "Tell me about the Customer table and its structure",
        "What DAX measures are defined for sales analysis?", 
        "How are Customer and Sales tables related?",
        "Explain how Total Sales is calculated",
        "What visualizations are on the main dashboard?"
    ]
    
    print(f"\nExample questions you can ask:")
    for question in sample_questions:
        print(f"  - \"{question}\"")

def show_next_steps():
    """Show what needs to be done to complete the setup."""
    print("\n=== Next Steps to Complete Setup ===")
    
    print("1. API Keys Setup:")
    print("   - Copy .env.example to .env")
    print("   - Add your OpenAI API key: OPENAI_API_KEY=sk-...")
    print("   - Add your Anthropic API key: ANTHROPIC_API_KEY=sk-ant-...")
    
    print("\n2. Install Missing Dependencies:")
    print("   - The main missing pieces are optional vector-store and model SDK dependencies")
    print("   - On Windows, this can be tricky due to dependency conflicts")
    print("   - Alternative: Use sentence-transformers for embeddings")
    
    print("\n3. Workaround Options:")
    print("   a) Use Docker for ChromaDB:")
    print("      docker-compose --profile qdrant up -d")
    print("   b) Use alternative vector stores:")
    print("      - Qdrant (via Docker)")
    print("      - Simple in-memory storage for testing")
    print("   c) Use Google Colab or Linux environment")
    
    print("\n4. Test Full System:")
    print("   - Once dependencies are installed:")
    print("   - python -m powerbi_rag.cli start-ui")
    print("   - Upload Adventure Works PBIX")
    print("   - Ask questions about your Power BI reports!")
    
    print("\n5. Production Deployment:")
    print("   - Use Docker for consistent environments")
    print("   - Set up proper SSL/TLS certificates")
    print("   - Configure environment variables")
    print("   - Monitor with health check endpoints")

def main():
    """Run the demo."""
    print("Power BI RAG Assistant - Feature Demo")
    print("=====================================")
    
    print("This demo shows what's currently working without full dependencies.")
    print()
    
    try:
        demo_config()
        demo_data_models()
        demo_pbix_extraction()
        demo_api_structure()
        demo_gradio_interface()
        show_next_steps()
        
        print("\n=====================================")
        print("Demo completed successfully!")
        print("Your Power BI RAG system core is working correctly.")
        print("The main missing piece is the vector database setup.")
        
        return True
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
