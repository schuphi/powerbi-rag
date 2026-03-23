"""Tests for PBIX extraction functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
from powerbi_rag.extraction.models import (
    PowerBIReport, PowerBIDataset, PowerBITable, 
    PowerBIMeasure, PowerBIColumn, ArtifactType
)


class TestPBIXExtractor:
    """Test PBIX extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = PBIXExtractor()
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'namespaces')
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_report("nonexistent.pbix")
    
    def test_invalid_file_extension(self):
        """Test handling of invalid file extensions."""
        # Create a temporary file with wrong extension
        temp_file = Path("test.txt")
        temp_file.write_text("test")
        
        try:
            with pytest.raises(ValueError, match="File must have .pbix extension"):
                self.extractor.extract_report(temp_file)
        finally:
            temp_file.unlink(missing_ok=True)
    
    @patch('zipfile.ZipFile')
    def test_extract_empty_pbix(self, mock_zipfile):
        """Test extraction of PBIX with minimal content."""
        # Mock ZIP file with no relevant content
        mock_zip = Mock()
        mock_zip.namelist.return_value = []
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Create temporary PBIX file
        temp_file = Path("test.pbix")
        temp_file.write_bytes(b"dummy content")
        
        try:
            report = self.extractor.extract_report(temp_file)
            
            assert isinstance(report, PowerBIReport)
            assert report.name == "test"
            assert isinstance(report.dataset, PowerBIDataset)
            assert report.dataset.name == "test"
            assert len(report.pages) == 0
            
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_create_table_artifacts(self):
        """Test creation of table artifacts."""
        # Create a sample table
        columns = [
            PowerBIColumn(
                name="CustomerID",
                data_type="int64",
                table_name="Customer",
                description="Unique customer identifier"
            )
        ]
        
        measures = [
            PowerBIMeasure(
                name="Total Sales",
                expression="SUM(Sales[Amount])",
                table_name="Customer",
                description="Total sales amount"
            )
        ]
        
        table = PowerBITable(
            name="Customer",
            description="Customer information",
            columns=columns,
            measures=measures
        )
        
        artifacts = self.extractor._create_table_artifacts(table, "TestReport")
        
        # Should create artifacts for table, columns, and measures
        assert len(artifacts) == 3  # 1 table + 1 column + 1 measure
        
        # Check table artifact
        table_artifact = artifacts[0]
        assert table_artifact.type == ArtifactType.TABLE
        assert table_artifact.name == "Customer"
        assert "1 columns and 1 measures" in table_artifact.content
        
        # Check column artifact
        column_artifact = next(a for a in artifacts if a.type == ArtifactType.COLUMN)
        assert column_artifact.name == "Customer.CustomerID"
        assert "int64" in column_artifact.content
        
        # Check measure artifact
        measure_artifact = next(a for a in artifacts if a.type == ArtifactType.MEASURE)
        assert measure_artifact.name == "Customer.Total Sales"
        assert "SUM(Sales[Amount])" in measure_artifact.content
    
    def test_artifact_id_generation(self):
        """Test that artifacts have unique IDs."""
        table = PowerBITable(
            name="Sales",
            columns=[
                PowerBIColumn(name="Amount", data_type="decimal", table_name="Sales"),
                PowerBIColumn(name="Date", data_type="datetime", table_name="Sales")
            ]
        )
        
        artifacts = self.extractor._create_table_artifacts(table, "TestReport")
        
        # Check all artifacts have unique IDs
        ids = [artifact.id for artifact in artifacts]
        assert len(ids) == len(set(ids))  # All IDs are unique
        
        # Check ID format
        for artifact in artifacts:
            assert artifact.id.startswith(f"{artifact.type}_")


@pytest.mark.integration
class TestPBIXExtractionIntegration:
    """Integration tests for PBIX extraction (requires sample files)."""
    
    @pytest.fixture
    def sample_pbix_path(self):
        """Provide path to sample PBIX file if available."""
        sample_path = Path("data/raw/Adventure Works Sales Sample.pbix")
        if not sample_path.exists():
            pytest.skip("Sample PBIX file not available. Run: python scripts/download_samples.py")
        return sample_path
    
    def test_extract_adventure_works(self, sample_pbix_path):
        """Test extraction of Adventure Works sample."""
        extractor = PBIXExtractor()
        
        try:
            report = extractor.extract_report(sample_pbix_path)
            
            # Basic validation
            assert isinstance(report, PowerBIReport)
            assert report.name == "Adventure Works Sales Sample"
            assert report.file_path == str(sample_pbix_path)
            
            # Dataset validation
            assert isinstance(report.dataset, PowerBIDataset)
            assert len(report.dataset.tables) > 0
            assert len(report.pages) > 0
            
            # Generate artifacts
            artifacts = extractor.extract_artifacts(report)
            assert len(artifacts) > 10
            
            # Check artifact types
            artifact_types = {artifact.type for artifact in artifacts}
            assert ArtifactType.REPORT in artifact_types
            assert ArtifactType.DATASET in artifact_types
            
            print(f"Extracted {len(artifacts)} artifacts from Adventure Works")
            print(f"Artifact types: {artifact_types}")
            
        except Exception as e:
            pytest.fail(f"Failed to extract Adventure Works sample: {e}")
    
    def test_artifact_content_quality(self, sample_pbix_path):
        """Test that extracted artifacts have meaningful content."""
        extractor = PBIXExtractor()
        report = extractor.extract_report(sample_pbix_path)
        artifacts = extractor.extract_artifacts(report)
        
        # Check that artifacts have meaningful content
        for artifact in artifacts[:10]:  # Check first 10
            assert len(artifact.content) > 10, f"Artifact {artifact.id} has minimal content"
            assert artifact.name, f"Artifact {artifact.id} has no name"
            assert artifact.type in ArtifactType, f"Invalid artifact type: {artifact.type}"
        
        # Check for expected Adventure Works entities
        artifact_names = [artifact.name.lower() for artifact in artifacts]
        
        # Should find some common Adventure Works entities
        expected_terms = ['customer', 'product', 'sales', 'territory']
        found_terms = [term for term in expected_terms 
                      if any(term in name for name in artifact_names)]
        
        assert len(found_terms) > 0, f"Expected Adventure Works entities not found in: {artifact_names[:10]}"
