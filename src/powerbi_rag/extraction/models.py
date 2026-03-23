"""Data models for Power BI extracted content."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ArtifactType(str, Enum):
    """Types of Power BI artifacts."""
    TABLE = "table"
    MEASURE = "measure"
    COLUMN = "column"
    RELATIONSHIP = "relationship"
    VISUAL = "visual"
    PAGE = "page"
    REPORT = "report"
    DATASET = "dataset"


class PowerBIColumn(BaseModel):
    """Represents a Power BI column."""
    name: str
    data_type: str
    table_name: str
    description: Optional[str] = None
    is_hidden: bool = False
    sort_by_column: Optional[str] = None
    format_string: Optional[str] = None


class PowerBIMeasure(BaseModel):
    """Represents a Power BI measure."""
    name: str
    expression: str
    table_name: str
    description: Optional[str] = None
    display_folder: Optional[str] = None
    format_string: Optional[str] = None
    is_hidden: bool = False


class PowerBITable(BaseModel):
    """Represents a Power BI table."""
    name: str
    description: Optional[str] = None
    columns: List[PowerBIColumn] = Field(default_factory=list)
    measures: List[PowerBIMeasure] = Field(default_factory=list)
    is_hidden: bool = False
    source_query: Optional[str] = None


class PowerBIRelationship(BaseModel):
    """Represents a Power BI relationship."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str
    cross_filter_direction: str
    is_active: bool = True
    security_filtering_behavior: Optional[str] = None


class PowerBIVisual(BaseModel):
    """Represents a Power BI visual."""
    name: str
    visual_type: str
    page_name: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    fields: List[str] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    position: Optional[Dict[str, float]] = None
    size: Optional[Dict[str, float]] = None


class PowerBIPage(BaseModel):
    """Represents a Power BI report page."""
    name: str
    display_name: Optional[str] = None
    visuals: List[PowerBIVisual] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    is_hidden: bool = False


class PowerBIDataset(BaseModel):
    """Represents a Power BI dataset."""
    name: str
    tables: List[PowerBITable] = Field(default_factory=list)
    relationships: List[PowerBIRelationship] = Field(default_factory=list)
    culture: Optional[str] = None
    compatibility_level: Optional[int] = None


class PowerBIReport(BaseModel):
    """Represents a complete Power BI report."""
    name: str
    dataset: PowerBIDataset
    pages: List[PowerBIPage] = Field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    version: Optional[str] = None
    file_path: Optional[str] = None


class PowerBIArtifact(BaseModel):
    """Represents a searchable Power BI artifact for RAG."""
    id: str = Field(description="Unique identifier for the artifact")
    type: ArtifactType
    name: str
    content: str = Field(description="Searchable content description")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_file: Optional[str] = None
    parent_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(use_enum_values=True)
