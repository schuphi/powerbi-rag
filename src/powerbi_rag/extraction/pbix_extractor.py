"""PBIX file extraction functionality."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import (
    PowerBIArtifact,
    PowerBIColumn,
    PowerBIDataset, 
    PowerBIMeasure,
    PowerBIPage,
    PowerBIRelationship,
    PowerBIReport,
    PowerBITable,
    PowerBIVisual,
    ArtifactType
)


class PBIXExtractor:
    """Extracts metadata and content from PBIX files."""
    
    def __init__(self):
        """Initialize the PBIX extractor."""
        self.namespaces = {
            'ds': 'http://schemas.microsoft.com/sqlserver/2003/12/dmschema/tabular',
            'bi': 'http://schemas.microsoft.com/sqlserver/2008/05/MiningStructure'
        }
    
    def extract_report(self, pbix_path: Union[str, Path]) -> PowerBIReport:
        """Extract complete report structure from PBIX file."""
        pbix_path = Path(pbix_path)
        
        if not pbix_path.exists():
            raise FileNotFoundError(f"PBIX file not found: {pbix_path}")
        
        if not pbix_path.suffix.lower() == '.pbix':
            raise ValueError(f"File must have .pbix extension: {pbix_path}")
        
        with zipfile.ZipFile(pbix_path, 'r') as zip_file:
            layout_data = self._load_layout_data(zip_file)
            diagram_data = self._load_diagram_data(zip_file)

            # Extract dataset metadata
            dataset = self._extract_dataset(zip_file, pbix_path.stem, layout_data, diagram_data)
            
            # Extract report pages and visuals
            pages = self._extract_pages(zip_file, layout_data)
            
            report = PowerBIReport(
                name=pbix_path.stem,
                dataset=dataset,
                pages=pages,
                file_path=str(pbix_path)
            )
            
        return report
    
    def _extract_dataset(
        self,
        zip_file: zipfile.ZipFile,
        report_name: str,
        layout_data: Optional[dict] = None,
        diagram_data: Optional[dict] = None
    ) -> PowerBIDataset:
        """Extract dataset information from DataModel file."""
        try:
            # Look for DataModel file (contains the tabular model definition)
            datamodel_content = None
            for file_name in zip_file.namelist():
                if file_name.endswith("DataModel.json") or file_name == "DataModelSchema":
                    datamodel_content = self._decode_text(zip_file.read(file_name))
                    break
            
            if datamodel_content:
                data = json.loads(datamodel_content)
                return self._parse_dataset_json(data)
            
            # Real-world PBIX files often store the semantic model as a binary DataModel.
            # Fall back to inferring tables/measures/columns from DiagramLayout and Report/Layout.
            fallback_dataset = self._infer_dataset_from_layout(
                report_name=report_name,
                layout_data=layout_data,
                diagram_data=diagram_data
            )
            if fallback_dataset:
                return fallback_dataset

            return PowerBIDataset(name=report_name)
            
        except Exception:
            fallback_dataset = self._infer_dataset_from_layout(
                report_name=report_name,
                layout_data=layout_data,
                diagram_data=diagram_data
            )
            return fallback_dataset or PowerBIDataset(name=report_name)
    
    def _parse_dataset_json(self, data: dict) -> PowerBIDataset:
        """Parse dataset from JSON data."""
        model = data.get('model', {})
        
        # Extract tables
        tables = []
        for table_data in model.get('tables', []):
            table = self._parse_table(table_data)
            tables.append(table)
        
        # Extract relationships
        relationships = []
        for rel_data in model.get('relationships', []):
            relationship = self._parse_relationship(rel_data)
            relationships.append(relationship)
        
        return PowerBIDataset(
            name=model.get('name', 'Dataset'),
            tables=tables,
            relationships=relationships,
            culture=model.get('culture'),
            compatibility_level=model.get('compatibilityLevel')
        )
    
    def _parse_table(self, table_data: dict) -> PowerBITable:
        """Parse table from JSON data."""
        # Extract columns
        columns = []
        for col_data in table_data.get('columns', []):
            column = PowerBIColumn(
                name=col_data.get('name', ''),
                data_type=col_data.get('dataType', 'unknown'),
                table_name=table_data.get('name', ''),
                description=col_data.get('description'),
                is_hidden=col_data.get('isHidden', False),
                sort_by_column=col_data.get('sortByColumn'),
                format_string=col_data.get('formatString')
            )
            columns.append(column)
        
        # Extract measures
        measures = []
        for measure_data in table_data.get('measures', []):
            measure = PowerBIMeasure(
                name=measure_data.get('name', ''),
                expression=measure_data.get('expression', ''),
                table_name=table_data.get('name', ''),
                description=measure_data.get('description'),
                display_folder=measure_data.get('displayFolder'),
                format_string=measure_data.get('formatString'),
                is_hidden=measure_data.get('isHidden', False)
            )
            measures.append(measure)
        
        return PowerBITable(
            name=table_data.get('name', ''),
            description=table_data.get('description'),
            columns=columns,
            measures=measures,
            is_hidden=table_data.get('isHidden', False),
            source_query=table_data.get('source', {}).get('expression')
        )
    
    def _parse_relationship(self, rel_data: dict) -> PowerBIRelationship:
        """Parse relationship from JSON data."""
        return PowerBIRelationship(
            from_table=rel_data.get('fromTable', ''),
            from_column=rel_data.get('fromColumn', ''),
            to_table=rel_data.get('toTable', ''),
            to_column=rel_data.get('toColumn', ''),
            cardinality=rel_data.get('cardinality', 'unknown'),
            cross_filter_direction=rel_data.get('crossFilteringBehavior', 'OneDirection'),
            is_active=rel_data.get('isActive', True)
        )
    
    def _extract_pages(
        self,
        zip_file: zipfile.ZipFile,
        layout_data: Optional[dict] = None
    ) -> List[PowerBIPage]:
        """Extract report pages from Report folder."""
        pages = []
        
        try:
            if layout_data and layout_data.get("sections"):
                for section in layout_data.get("sections", []):
                    page = self._parse_section(section)
                    if page:
                        pages.append(page)
                return pages

            # Compatibility path for old JSON-per-page formats.
            for file_name in zip_file.namelist():
                if 'Report/Layout' in file_name:
                    if layout_data is None:
                        layout_data = self._load_json_entry(zip_file, file_name)
                    if layout_data:
                        page = self._parse_page(layout_data, file_name)
                        if page:
                            pages.append(page)
        
        except Exception:
            # Fallback for extraction errors
            pass
            
        return pages
    
    def _parse_page(self, layout_data: dict, file_name: str) -> Optional[PowerBIPage]:
        """Parse page from layout JSON data."""
        try:
            # Extract page information
            page_name = layout_data.get('name', Path(file_name).stem)
            display_name = layout_data.get('displayName', page_name)
            
            # Extract visuals
            visuals = []
            for visual_data in layout_data.get('visuals', []):
                visual = self._parse_visual(visual_data, page_name)
                if visual:
                    visuals.append(visual)
            
            return PowerBIPage(
                name=page_name,
                display_name=display_name,
                visuals=visuals,
                is_hidden=layout_data.get('isHidden', False)
            )
            
        except Exception:
            return None
    
    def _parse_visual(self, visual_data: dict, page_name: str) -> Optional[PowerBIVisual]:
        """Parse visual from JSON data."""
        try:
            config = visual_data.get('config', {})
            
            return PowerBIVisual(
                name=visual_data.get('name', ''),
                visual_type=config.get('singleVisual', {}).get('visualType', 'unknown'),
                page_name=page_name,
                title=config.get('vcObjects', {}).get('title', [{}])[0].get('properties', {}).get('text', {}).get('expr', {}).get('Literal', {}).get('Value'),
                fields=self._extract_visual_fields(config),
                position=visual_data.get('position'),
                size=visual_data.get('size')
            )
            
        except Exception:
            return None

    def _parse_section(self, section_data: dict) -> Optional[PowerBIPage]:
        """Parse a report section from Report/Layout."""
        try:
            page_name = section_data.get("name") or f"Section{section_data.get('ordinal', 0)}"
            display_name = section_data.get("displayName") or page_name

            visuals = []
            for visual_data in section_data.get("visualContainers", []):
                visual = self._parse_visual_container(visual_data, page_name)
                if visual:
                    visuals.append(visual)

            return PowerBIPage(
                name=page_name,
                display_name=display_name,
                visuals=visuals,
                filters=self._parse_jsonish(section_data.get("filters")) or [],
                is_hidden=section_data.get("isHidden", False),
            )
        except Exception:
            return None

    def _parse_visual_container(self, visual_data: dict, page_name: str) -> Optional[PowerBIVisual]:
        """Parse a visual container from Report/Layout."""
        try:
            config = self._parse_jsonish(visual_data.get("config")) or {}
            single_visual = config.get("singleVisual", {})
            position = {
                "x": visual_data.get("x"),
                "y": visual_data.get("y"),
                "z": visual_data.get("z"),
            }
            size = {
                "width": visual_data.get("width"),
                "height": visual_data.get("height"),
            }

            return PowerBIVisual(
                name=config.get("name", ""),
                visual_type=single_visual.get("visualType", "unknown"),
                page_name=page_name,
                title=self._extract_visual_title(config),
                fields=self._extract_visual_fields(config),
                filters=self._parse_jsonish(visual_data.get("filters")) or [],
                position={k: v for k, v in position.items() if v is not None} or None,
                size={k: v for k, v in size.items() if v is not None} or None,
            )
        except Exception:
            return None
    
    def _extract_visual_fields(self, config: dict) -> List[str]:
        """Extract field names used in a visual."""
        fields = []
        
        try:
            # Navigate through the config structure to find field references
            single_visual = config.get('singleVisual', {})
            projections = single_visual.get('projections', {})
            
            for projection_key, projection_data in projections.items():
                if isinstance(projection_data, list):
                    for item in projection_data:
                        if 'queryRef' in item:
                            fields.append(item['queryRef'])
                        elif 'selectItems' in item:
                            for select_item in item['selectItems']:
                                if 'displayName' in select_item:
                                    fields.append(select_item['displayName'])

            prototype_query = single_visual.get("prototypeQuery", {})
            for query_ref in self._collect_query_refs(prototype_query):
                fields.append(query_ref)
                        
        except Exception:
            pass
            
        return list(dict.fromkeys(fields))
    
    def extract_artifacts(self, report: PowerBIReport) -> List[PowerBIArtifact]:
        """Convert report structure to searchable artifacts."""
        artifacts = []
        
        # Report-level artifact
        artifacts.append(PowerBIArtifact(
            id=f"report_{report.name}",
            type=ArtifactType.REPORT,
            name=report.name,
            content=f"Power BI Report: {report.name}. Contains {len(report.dataset.tables)} tables and {len(report.pages)} pages.",
            metadata={"file_path": report.file_path},
            source_file=report.file_path
        ))
        
        # Dataset-level artifact
        artifacts.append(self._create_dataset_artifact(report.dataset, report.name, report.file_path))
        
        # Table artifacts
        for table in report.dataset.tables:
            artifacts.extend(self._create_table_artifacts(table, report.name, report.file_path))
        
        # Relationship artifacts
        for relationship in report.dataset.relationships:
            artifacts.append(self._create_relationship_artifact(relationship, report.name, report.file_path))
        
        # Page and visual artifacts
        for page in report.pages:
            artifacts.extend(self._create_page_artifacts(page, report.name, report.file_path))
        
        return artifacts
    
    def _create_dataset_artifact(
        self,
        dataset: PowerBIDataset,
        report_name: str,
        source_file: Optional[str] = None
    ) -> PowerBIArtifact:
        """Create artifact for dataset."""
        content = f"Dataset: {dataset.name}. Contains {len(dataset.tables)} tables and {len(dataset.relationships)} relationships."
        if dataset.culture:
            content += f" Culture: {dataset.culture}."
            
        return PowerBIArtifact(
            id=f"dataset_{dataset.name}",
            type=ArtifactType.DATASET,
            name=dataset.name,
            content=content,
            metadata={
                "table_count": len(dataset.tables),
                "relationship_count": len(dataset.relationships),
                "culture": dataset.culture
            },
            source_file=source_file,
            parent_id=f"report_{report_name}",
            tags=["dataset", "data-model"]
        )
    
    def _create_table_artifacts(
        self,
        table: PowerBITable,
        report_name: str,
        source_file: Optional[str] = None
    ) -> List[PowerBIArtifact]:
        """Create artifacts for table and its components."""
        artifacts = []
        
        # Table artifact
        content = f"Table: {table.name}. Contains {len(table.columns)} columns and {len(table.measures)} measures."
        if table.description:
            content += f" Description: {table.description}"
            
        artifacts.append(PowerBIArtifact(
            id=f"table_{table.name}",
            type=ArtifactType.TABLE,
            name=table.name,
            content=content,
            metadata={
                "column_count": len(table.columns),
                "measure_count": len(table.measures),
                "is_hidden": table.is_hidden
            },
            source_file=source_file,
            parent_id=f"dataset_{report_name}",
            tags=["table", "data-structure"]
        ))
        
        # Column artifacts
        for column in table.columns:
            artifacts.append(self._create_column_artifact(column, table.name, source_file))
        
        # Measure artifacts  
        for measure in table.measures:
            artifacts.append(self._create_measure_artifact(measure, table.name, source_file))
            
        return artifacts
    
    def _create_column_artifact(
        self,
        column: PowerBIColumn,
        table_name: str,
        source_file: Optional[str] = None
    ) -> PowerBIArtifact:
        """Create artifact for column."""
        content = f"Column: {column.name} in table {column.table_name}. Data type: {column.data_type}."
        if column.description:
            content += f" Description: {column.description}"
        if column.format_string:
            content += f" Format: {column.format_string}"
            
        return PowerBIArtifact(
            id=f"column_{table_name}_{column.name}",
            type=ArtifactType.COLUMN,
            name=f"{table_name}.{column.name}",
            content=content,
            metadata={
                "data_type": column.data_type,
                "table_name": column.table_name,
                "is_hidden": column.is_hidden,
                "format_string": column.format_string
            },
            source_file=source_file,
            parent_id=f"table_{table_name}",
            tags=["column", "field", column.data_type.lower()]
        )
    
    def _create_measure_artifact(
        self,
        measure: PowerBIMeasure,
        table_name: str,
        source_file: Optional[str] = None
    ) -> PowerBIArtifact:
        """Create artifact for measure."""
        content = f"Measure: {measure.name} in table {measure.table_name}. DAX Expression: {measure.expression}"
        if measure.description:
            content += f" Description: {measure.description}"
        if measure.display_folder:
            content += f" Display Folder: {measure.display_folder}"
            
        return PowerBIArtifact(
            id=f"measure_{table_name}_{measure.name}",
            type=ArtifactType.MEASURE,
            name=f"{table_name}.{measure.name}",
            content=content,
            metadata={
                "expression": measure.expression,
                "table_name": measure.table_name,
                "display_folder": measure.display_folder,
                "is_hidden": measure.is_hidden,
                "format_string": measure.format_string
            },
            source_file=source_file,
            parent_id=f"table_{table_name}",
            tags=["measure", "dax", "calculation"]
        )
    
    def _create_relationship_artifact(
        self,
        relationship: PowerBIRelationship,
        report_name: str,
        source_file: Optional[str] = None
    ) -> PowerBIArtifact:
        """Create artifact for relationship."""
        content = f"Relationship: {relationship.from_table}.{relationship.from_column} -> {relationship.to_table}.{relationship.to_column}. Cardinality: {relationship.cardinality}, Cross-filter: {relationship.cross_filter_direction}"
        
        return PowerBIArtifact(
            id=f"relationship_{relationship.from_table}_{relationship.to_table}",
            type=ArtifactType.RELATIONSHIP,
            name=f"{relationship.from_table} -> {relationship.to_table}",
            content=content,
            metadata={
                "from_table": relationship.from_table,
                "from_column": relationship.from_column,
                "to_table": relationship.to_table,
                "to_column": relationship.to_column,
                "cardinality": relationship.cardinality,
                "cross_filter_direction": relationship.cross_filter_direction,
                "is_active": relationship.is_active
            },
            source_file=source_file,
            parent_id=f"dataset_{report_name}",
            tags=["relationship", "data-model"]
        )
    
    def _create_page_artifacts(
        self,
        page: PowerBIPage,
        report_name: str,
        source_file: Optional[str] = None
    ) -> List[PowerBIArtifact]:
        """Create artifacts for page and its visuals."""
        artifacts = []
        
        # Page artifact
        content = f"Report Page: {page.display_name or page.name}. Contains {len(page.visuals)} visuals."
        
        artifacts.append(PowerBIArtifact(
            id=f"page_{page.name}",
            type=ArtifactType.PAGE,
            name=page.display_name or page.name,
            content=content,
            metadata={
                "visual_count": len(page.visuals),
                "is_hidden": page.is_hidden
            },
            source_file=source_file,
            parent_id=f"report_{report_name}",
            tags=["page", "report-page"]
        ))
        
        # Visual artifacts
        for visual in page.visuals:
            artifacts.append(self._create_visual_artifact(visual, page.name, source_file))
            
        return artifacts
    
    def _create_visual_artifact(
        self,
        visual: PowerBIVisual,
        page_name: str,
        source_file: Optional[str] = None
    ) -> PowerBIArtifact:
        """Create artifact for visual."""
        content = f"Visual: {visual.name or 'Unnamed'} of type {visual.visual_type} on page {visual.page_name}."
        if visual.title:
            content += f" Title: {visual.title}."
        if visual.fields:
            content += f" Uses fields: {', '.join(visual.fields)}."
            
        return PowerBIArtifact(
            id=f"visual_{page_name}_{visual.name or 'unnamed'}",
            type=ArtifactType.VISUAL,
            name=visual.title or visual.name or f"{visual.visual_type} Visual",
            content=content,
            metadata={
                "visual_type": visual.visual_type,
                "page_name": visual.page_name,
                "fields": visual.fields,
                "position": visual.position,
                "size": visual.size
            },
            source_file=source_file,
            parent_id=f"page_{page_name}",
            tags=["visual", visual.visual_type.lower().replace(" ", "-")]
        )

    def _load_layout_data(self, zip_file: zipfile.ZipFile) -> Optional[dict]:
        """Load Report/Layout if available."""
        for file_name in zip_file.namelist():
            if file_name == "Report/Layout":
                return self._load_json_entry(zip_file, file_name)
        return None

    def _load_diagram_data(self, zip_file: zipfile.ZipFile) -> Optional[dict]:
        """Load DiagramLayout if available."""
        if "DiagramLayout" in zip_file.namelist():
            return self._load_json_entry(zip_file, "DiagramLayout")
        return None

    def _load_json_entry(self, zip_file: zipfile.ZipFile, file_name: str) -> Optional[dict]:
        """Load a UTF-8 or UTF-16 encoded JSON-like zip entry."""
        try:
            return json.loads(self._decode_text(zip_file.read(file_name)))
        except Exception:
            return None

    def _decode_text(self, data: bytes) -> str:
        """Decode PBIX text entries that are commonly UTF-16-LE encoded."""
        encodings = ("utf-16-le", "utf-16", "utf-8-sig", "utf-8") if b"\x00" in data else ("utf-8-sig", "utf-8", "utf-16-le", "utf-16")
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("latin1")

    def _parse_jsonish(self, value: Any) -> Any:
        """Parse strings that contain JSON payloads."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _extract_visual_title(self, config: dict) -> Optional[str]:
        """Extract a visual title from the config structure."""
        try:
            title_value = (
                config.get("singleVisual", {})
                .get("vcObjects", {})
                .get("title", [{}])[0]
                .get("properties", {})
                .get("text", {})
                .get("expr", {})
                .get("Literal", {})
                .get("Value")
            )
            if isinstance(title_value, str):
                return title_value.strip("'\"")
        except Exception:
            pass
        return None

    def _collect_query_refs(self, value: Any) -> List[str]:
        """Collect queryRef strings recursively from nested config/query payloads."""
        refs: List[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "queryRef" and isinstance(item, str):
                    refs.append(item)
                refs.extend(self._collect_query_refs(item))
        elif isinstance(value, list):
            for item in value:
                refs.extend(self._collect_query_refs(item))
        return refs

    def _infer_dataset_from_layout(
        self,
        report_name: str,
        layout_data: Optional[dict],
        diagram_data: Optional[dict]
    ) -> Optional[PowerBIDataset]:
        """Infer dataset structure from report layout/diagram metadata."""
        table_info: Dict[str, Dict[str, set[str]]] = {}

        def ensure_table(table_name: str) -> Dict[str, set[str]]:
            return table_info.setdefault(table_name, {"columns": set(), "measures": set()})

        if diagram_data:
            for diagram in diagram_data.get("diagrams", []):
                for node in diagram.get("nodes", []):
                    node_index = node.get("nodeIndex")
                    if node_index:
                        ensure_table(node_index)

        if layout_data:
            for section in layout_data.get("sections", []):
                for visual_data in section.get("visualContainers", []):
                    config = self._parse_jsonish(visual_data.get("config"))
                    if not isinstance(config, dict):
                        continue
                    tables, columns, measures = self._extract_semantic_entities(config)
                    for table_name in tables:
                        ensure_table(table_name)
                    for table_name, column_name in columns:
                        ensure_table(table_name)["columns"].add(column_name)
                    for table_name, measure_name in measures:
                        ensure_table(table_name)["measures"].add(measure_name)

        if not table_info:
            return None

        tables = []
        for table_name in sorted(table_info):
            info = table_info[table_name]
            columns = [
                PowerBIColumn(name=name, data_type="unknown", table_name=table_name)
                for name in sorted(info["columns"])
            ]
            measures = [
                PowerBIMeasure(name=name, expression="", table_name=table_name)
                for name in sorted(info["measures"])
            ]
            tables.append(
                PowerBITable(
                    name=table_name,
                    columns=columns,
                    measures=measures,
                )
            )

        return PowerBIDataset(name=report_name, tables=tables, relationships=[])

    def _extract_semantic_entities(
        self,
        config: dict
    ) -> Tuple[set[str], set[Tuple[str, str]], set[Tuple[str, str]]]:
        """Extract tables, columns, and measures from a visual config."""
        tables: set[str] = set()
        columns: set[Tuple[str, str]] = set()
        measures: set[Tuple[str, str]] = set()

        single_visual = config.get("singleVisual", {})
        prototype_query = single_visual.get("prototypeQuery", {})
        alias_to_table = {
            entry.get("Name"): entry.get("Entity")
            for entry in prototype_query.get("From", [])
            if entry.get("Name") and entry.get("Entity")
        }
        tables.update(alias_to_table.values())

        for select_item in prototype_query.get("Select", []):
            parsed_table, parsed_name, parsed_kind = self._parse_select_item(select_item, alias_to_table)
            if parsed_table and parsed_name:
                tables.add(parsed_table)
                if parsed_kind == "measure":
                    measures.add((parsed_table, parsed_name))
                else:
                    columns.add((parsed_table, parsed_name))

            name = select_item.get("Name")
            table_name, field_name = self._split_query_ref(name)
            if table_name and field_name:
                tables.add(table_name)
                target = measures if self._looks_like_measure_ref(name) else columns
                target.add((table_name, field_name))

        for query_ref in self._collect_query_refs(single_visual):
            table_name, field_name = self._split_query_ref(query_ref)
            if table_name and field_name:
                tables.add(table_name)
                target = measures if self._looks_like_measure_ref(query_ref) else columns
                target.add((table_name, field_name))

        return tables, columns, measures

    def _parse_select_item(
        self,
        select_item: dict,
        alias_to_table: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a prototypeQuery select item into (table, field, kind)."""
        if "Measure" in select_item:
            measure = select_item["Measure"]
            source = measure.get("Expression", {}).get("SourceRef", {}).get("Source")
            table_name = alias_to_table.get(source)
            return table_name, measure.get("Property"), "measure"

        if "Column" in select_item:
            column = select_item["Column"]
            source = column.get("Expression", {}).get("SourceRef", {}).get("Source")
            table_name = alias_to_table.get(source)
            return table_name, column.get("Property"), "column"

        if "Aggregation" in select_item:
            column = (
                select_item["Aggregation"]
                .get("Expression", {})
                .get("Column", {})
            )
            source = column.get("Expression", {}).get("SourceRef", {}).get("Source")
            table_name = alias_to_table.get(source)
            return table_name, column.get("Property"), "column"

        if "HierarchyLevel" in select_item:
            hierarchy = select_item["HierarchyLevel"].get("Expression", {}).get("Hierarchy", {})
            source = hierarchy.get("Expression", {}).get("SourceRef", {}).get("Source")
            table_name = alias_to_table.get(source)
            hierarchy_name = hierarchy.get("Hierarchy")
            level_name = select_item["HierarchyLevel"].get("Level")
            field_name = ".".join(part for part in [hierarchy_name, level_name] if part)
            return table_name, field_name or None, "column"

        return None, None, None

    def _split_query_ref(self, query_ref: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Split Power BI query refs like Table.Field or Sum(Table.Field)."""
        if not query_ref or not isinstance(query_ref, str):
            return None, None

        if query_ref.startswith("Sum(") and query_ref.endswith(")"):
            query_ref = query_ref[4:-1]

        if "." not in query_ref:
            return None, None

        table_name, field_name = query_ref.split(".", 1)
        return table_name or None, field_name or None

    def _looks_like_measure_ref(self, query_ref: Optional[str]) -> bool:
        """Heuristic to distinguish measure refs from raw field refs."""
        if not query_ref or not isinstance(query_ref, str):
            return False
        return query_ref.startswith("Sum(") or " by " in query_ref
