"""Command line interface for Power BI RAG Assistant."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .extraction.pbix_extractor import PBIXExtractor
from .utils.config import settings

app = typer.Typer(name="powerbi-rag", help="Power BI RAG Assistant CLI")
console = Console()
HELP_TEXT = """Usage: python -m powerbi_rag.cli COMMAND [ARGS]...

Commands:
  extract-pbix FILE_PATH    Extract metadata and artifacts from a PBIX file
  list-files                List available PBIX files in the data directory
  start-api                 Start the FastAPI server
  start-ui                  Start the Gradio web interface
  config                    Show configuration values
"""


@app.command()
def extract_pbix(
    file_path: str = typer.Argument(..., help="Path to PBIX file"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for extracted data"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, artifacts)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Extract metadata and structure from a PBIX file."""
    
    pbix_path = Path(file_path)
    if not pbix_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = f"./data/processed/{pbix_path.stem}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[blue]Extracting PBIX file: {pbix_path.name}[/blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Initialize extractor
            task = progress.add_task("Initializing extractor...", total=None)
            extractor = PBIXExtractor()
            
            # Extract report structure
            progress.update(task, description="Extracting report structure...")
            report = extractor.extract_report(pbix_path)
            
            # Generate artifacts
            progress.update(task, description="Generating searchable artifacts...")
            artifacts = extractor.extract_artifacts(report)
            
            # Save results
            progress.update(task, description="Saving results...")
            if format == "json":
                # Save full report structure
                report_file = output_path / "report.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report.model_dump(), f, indent=2, default=str)
                console.print(f"[green]Report structure saved to: {report_file}[/green]")
                
            elif format == "artifacts":
                # Save artifacts for RAG
                artifacts_file = output_path / "artifacts.json"
                artifacts_data = [artifact.model_dump() for artifact in artifacts]
                with open(artifacts_file, 'w', encoding='utf-8') as f:
                    json.dump(artifacts_data, f, indent=2, default=str)
                console.print(f"[green]Artifacts saved to: {artifacts_file}[/green]")
            
            progress.update(task, description="Complete!")
        
        # Display summary
        _display_extraction_summary(report, artifacts, verbose)
        
    except Exception as e:
        console.print(f"[red]Error during extraction: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_extraction_summary(report, artifacts, verbose: bool):
    """Display extraction summary."""
    console.print("\n[bold blue]Extraction Summary[/bold blue]")
    
    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="dim")
    table.add_column("Count", justify="right")
    
    table.add_row("Tables", str(len(report.dataset.tables)))
    table.add_row("Relationships", str(len(report.dataset.relationships)))
    table.add_row("Report Pages", str(len(report.pages)))
    
    # Count columns and measures across all tables
    total_columns = sum(len(table.columns) for table in report.dataset.tables)
    total_measures = sum(len(table.measures) for table in report.dataset.tables)
    total_visuals = sum(len(page.visuals) for page in report.pages)
    
    table.add_row("Columns", str(total_columns))
    table.add_row("Measures", str(total_measures))
    table.add_row("Visuals", str(total_visuals))
    table.add_row("Total Artifacts", str(len(artifacts)))
    
    console.print(table)
    
    if verbose:
        console.print("\n[bold blue]Detailed Information[/bold blue]")
        
        # Tables
        if report.dataset.tables:
            console.print("\n[bold]Tables:[/bold]")
            for table in report.dataset.tables:
                console.print(f"  • {table.name} ({len(table.columns)} columns, {len(table.measures)} measures)")
        
        # Pages
        if report.pages:
            console.print("\n[bold]Pages:[/bold]")
            for page in report.pages:
                console.print(f"  • {page.display_name or page.name} ({len(page.visuals)} visuals)")


@app.command()
def list_files(
    data_dir: str = typer.Option("./data/raw", "--dir", "-d", help="Data directory to scan")
):
    """List available PBIX files in data directory."""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)
    
    pbix_files = list(data_path.glob("*.pbix"))
    
    if not pbix_files:
        console.print(f"[yellow]No PBIX files found in: {data_dir}[/yellow]")
        return
    
    console.print(f"[blue]PBIX files in {data_dir}:[/blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Modified", justify="right")
    
    for file in sorted(pbix_files):
        stat = file.stat()
        size = f"{stat.st_size / (1024*1024):.1f} MB"
        modified = f"{stat.st_mtime}"
        table.add_row(file.name, size, modified)
    
    console.print(table)


@app.command()
def start_api(
    host: str = typer.Option(None, "--host", help="API host"),
    port: int = typer.Option(None, "--port", help="API port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload")
):
    """Start the FastAPI server."""
    import uvicorn
    
    api_host = host or settings.api.host
    api_port = port or settings.api.port
    
    console.print(f"[blue]Starting API server on http://{api_host}:{api_port}[/blue]")
    
    try:
        uvicorn.run(
            "powerbi_rag.api.main:app",
            host=api_host,
            port=api_port,
            reload=reload,
            log_level=settings.log_level.lower()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]API server stopped[/yellow]")


@app.command()
def start_ui(
    host: str = typer.Option(None, "--host", help="UI host"),
    port: int = typer.Option(None, "--port", help="UI port"),
    share: bool = typer.Option(None, "--share", help="Create public link")
):
    """Start the Gradio web interface."""
    from .ui.gradio_app import create_app
    
    ui_host = host or settings.ui.host
    ui_port = port or settings.ui.port
    ui_share = share if share is not None else settings.ui.share
    
    console.print(f"[blue]Starting web interface on http://{ui_host}:{ui_port}[/blue]")
    
    try:
        app = create_app()
        app.launch(
            server_name=ui_host,
            server_port=ui_port,
            share=ui_share
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Web interface stopped[/yellow]")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    key: Optional[str] = typer.Option(None, "--key", help="Show specific config key")
):
    """Manage configuration settings."""
    
    if show or key:
        console.print("[blue]Current Configuration:[/blue]\n")
        
        if key:
            # Show specific key
            value = getattr(settings, key, "Not found")
            console.print(f"{key}: {value}")
        else:
            # Show all configuration
            config_dict = settings.model_dump()
            console.print_json(json.dumps(config_dict, indent=2, default=str))
    else:
        console.print("[yellow]Use --show to display current configuration[/yellow]")


def main():
    """Main CLI entry point."""
    if len(sys.argv) == 1 or any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        console.print(HELP_TEXT)
        return
    app()


if __name__ == "__main__":
    main()
