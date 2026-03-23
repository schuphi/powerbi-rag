"""Gradio web interface for Power BI RAG Assistant."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from ..extraction.pbix_extractor import PBIXExtractor
from ..processing.embeddings import EmbeddingProcessor
from ..retrieval.rag_pipeline import PowerBIRAGPipeline, ConversationManager
from ..retrieval.vector_store import ChromaVectorStore
from ..utils.config import settings


class PowerBIRAGInterface:
    """Gradio interface for Power BI RAG Assistant."""
    
    def __init__(self):
        """Initialize the interface."""
        self.vector_store = None
        self.rag_pipeline = None
        self.conversation_manager = None
        self.embedding_processor = None
        self.conversation_history = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components."""
        try:
            # Initialize vector store
            self.vector_store = ChromaVectorStore(
                embedding_function="openai" if settings.openai_api_key else "sentence_transformers"
            )
            
            # Initialize RAG pipeline
            self.rag_pipeline = PowerBIRAGPipeline(vector_store=self.vector_store)
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(self.rag_pipeline)
            
            # Initialize embedding processor
            self.embedding_processor = EmbeddingProcessor(
                provider="openai" if settings.openai_api_key else "sentence_transformers"
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize some components: {e}")
    
    def upload_and_process_pbix(
        self,
        file_path: str,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, str]:
        """Upload and process a PBIX file."""
        
        if not file_path or not Path(file_path).exists():
            return "No file selected or file not found.", ""
        
        if not file_path.lower().endswith('.pbix'):
            return "File must be a PBIX file.", ""
        
        try:
            progress(0, desc="Starting PBIX processing...")
            
            # Extract PBIX content
            progress(0.2, desc="Extracting PBIX metadata...")
            extractor = PBIXExtractor()
            report = extractor.extract_report(file_path)
            
            progress(0.4, desc="Generating searchable artifacts...")
            artifacts = extractor.extract_artifacts(report)
            
            if not artifacts:
                return "No artifacts extracted from PBIX file.", ""
            
            # Add to retrieval backends.
            progress(0.6, desc="Indexing artifacts for retrieval...")
            if self.rag_pipeline:
                self.rag_pipeline.index_artifacts(artifacts)
                progress(0.9)
            elif self.vector_store:
                batch_size = 50
                for i in range(0, len(artifacts), batch_size):
                    batch = artifacts[i:i + batch_size]
                    self.vector_store.add_artifacts(batch)
                    progress(0.6 + 0.3 * (i + len(batch)) / len(artifacts))
            
            progress(1.0, desc="Processing complete!")
            
            # Generate summary
            summary = self._generate_processing_summary(report, artifacts)
            
            success_msg = f"Successfully processed {Path(file_path).name}\\n{len(artifacts)} artifacts added to knowledge base."
            
            return success_msg, summary
            
        except Exception as e:
            return f"Error processing file: {str(e)}", ""
    
    def _generate_processing_summary(self, report, artifacts) -> str:
        """Generate a summary of the processed report."""
        
        # Count artifacts by type
        artifact_counts = {}
        for artifact in artifacts:
            artifact_type = artifact.type
            artifact_counts[artifact_type] = artifact_counts.get(artifact_type, 0) + 1
        
        # Generate summary text
        summary_parts = [
            f"**Report:** {report.name}",
            f"**Total Artifacts:** {len(artifacts)}",
            "",
            "**Artifact Breakdown:**"
        ]
        
        for artifact_type, count in sorted(artifact_counts.items()):
            summary_parts.append(f"- {artifact_type.title()}: {count}")
        
        # Add dataset info
        if report.dataset:
            summary_parts.extend([
                "",
                "**Dataset Information:**",
                f"- Tables: {len(report.dataset.tables)}",
                f"- Relationships: {len(report.dataset.relationships)}"
            ])
            
            if report.dataset.tables:
                total_columns = sum(len(table.columns) for table in report.dataset.tables)
                total_measures = sum(len(table.measures) for table in report.dataset.tables)
                summary_parts.extend([
                    f"- Columns: {total_columns}",
                    f"- Measures: {total_measures}"
                ])
        
        # Add pages info
        if report.pages:
            total_visuals = sum(len(page.visuals) for page in report.pages)
            summary_parts.extend([
                "",
                "**Report Pages:**",
                f"- Pages: {len(report.pages)}",
                f"- Visuals: {total_visuals}"
            ])
        
        return "\\n".join(summary_parts)
    
    async def ask_question_async(
        self,
        question: str,
        session_id: str,
        use_conversation_history: bool,
        filter_type: str
    ) -> Tuple[str, str, str]:
        """Ask a question (async version)."""
        
        if not question.strip():
            return "Please enter a question.", "", ""
        
        if not self.rag_pipeline:
            return "RAG system not initialized. Please check configuration.", "", ""
        
        try:
            # Get response
            if use_conversation_history and self.conversation_manager:
                filter_by_type = filter_type if filter_type != "All" else None
                response = await self.conversation_manager.ask_question(
                    question,
                    session_id=session_id,
                    use_history=True,
                    filter_by_type=filter_by_type,
                )
            else:
                filter_by_type = filter_type if filter_type != "All" else None
                response = await self.rag_pipeline.answer_question(
                    question,
                    filter_by_type=filter_by_type
                )
            
            # Format context information
            context_info = self._format_context_info(response)
            
            # Update conversation history display
            history_display = self._update_conversation_display(session_id, question, response["answer"])
            
            return response["answer"], context_info, history_display
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg, "", ""
    
    def ask_question(
        self,
        question: str,
        session_id: str = "default",
        use_conversation_history: bool = True,
        filter_type: str = "All"
    ) -> Tuple[str, str, str]:
        """Ask a question (sync wrapper)."""
        
        # Run async function in event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.ask_question_async(question, session_id, use_conversation_history, filter_type)
            )
            loop.close()
            return result
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def _format_context_info(self, response) -> str:
        """Format context information for display."""
        
        context_parts = [
            f"**Confidence:** {response.get('confidence', 0):.2f}",
            f"**Sources Found:** {len(response.get('context', []))}",
            ""
        ]
        
        if response.get('sources'):
            context_parts.append("**Sources:**")
            for source in response['sources'][:5]:  # Limit to 5 sources
                context_parts.append(f"- {source}")
            context_parts.append("")
        
        if response.get('context'):
            context_parts.append("**Retrieved Context:**")
            for i, ctx in enumerate(response['context'][:3], 1):  # Show top 3
                context_parts.append(f"{i}. **{ctx.get('type', '').title()}**: {ctx.get('name', 'Unknown')}")
                context_parts.append(f"   Score: {ctx.get('score', 0):.3f}")
                context_parts.append(f"   Content: {ctx.get('content', '')[:100]}...")
                context_parts.append("")
        
        return "\\n".join(context_parts)
    
    def _update_conversation_display(self, session_id: str, question: str, answer: str) -> str:
        """Update conversation history display."""
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Keep only last 10 interactions
        self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
        # Format for display
        history_parts = []
        for i, interaction in enumerate(self.conversation_history[session_id], 1):
            history_parts.extend([
                f"**Q{i}:** {interaction['question']}",
                f"**A{i}:** {interaction['answer'][:200]}{'...' if len(interaction['answer']) > 200 else ''}",
                ""
            ])
        
        return "\\n".join(history_parts)
    
    def clear_conversation(self, session_id: str = "default") -> str:
        """Clear conversation history."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
        
        if self.conversation_manager:
            self.conversation_manager.clear_history(session_id)
        
        return "Conversation history cleared."
    
    def get_system_status(self) -> str:
        """Get system status information."""
        
        status_parts = ["## System Status\\n"]
        
        # Component status
        components = {
            "Vector Store": "Available" if self.vector_store else "Unavailable",
            "RAG Pipeline": "Available" if self.rag_pipeline else "Unavailable",
            "Embedding Processor": "Available" if self.embedding_processor else "Unavailable",
            "Conversation Manager": "Available" if self.conversation_manager else "Unavailable"
        }
        
        status_parts.append("**Components:**")
        for component, status in components.items():
            status_parts.append(f"- {component}: {status}")
        
        # Vector store info
        if self.vector_store:
            try:
                info = self.vector_store.get_collection_info()
                status_parts.extend([
                    "",
                    "**Vector Store:**",
                    f"- Artifacts: {info.get('count', 0)}",
                    f"- Types: {', '.join(info.get('artifact_types', []))}",
                    f"- Embedding: {info.get('embedding_function', 'Unknown')}"
                ])
            except Exception as e:
                status_parts.append(f"- Error getting vector store info: {e}")
        
        # Configuration
        status_parts.extend([
            "",
            "**Configuration:**",
            f"- Environment: {settings.environment}",
            f"- LLM Model: {settings.llm_model}",
            f"- Embedding Model: {settings.embedding_model}",
            f"- Caching: {'Enabled' if settings.enable_caching else 'Disabled'}"
        ])
        
        return "\\n".join(status_parts)


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    
    # Initialize interface
    interface = PowerBIRAGInterface()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .status-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title="Power BI RAG Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        gr.Markdown("""
        # Power BI RAG Assistant
        
        Upload your Power BI reports and ask questions about tables, measures, relationships, and visualizations.
        """)
        
        with gr.Tabs():
            
            # File Upload Tab
            with gr.Tab("Upload Report"):
                gr.Markdown("### Upload and Process PBIX Files")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Select PBIX File",
                            file_types=[".pbix"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("Process PBIX File", variant="primary", size="lg")
                    
                    with gr.Column(scale=3):
                        upload_status = gr.Textbox(
                            label="Status",
                            placeholder="Select a PBIX file and click 'Process PBIX File'",
                            lines=3,
                            interactive=False
                        )
                
                processing_summary = gr.Markdown(
                    label="Processing Summary",
                    value="Upload a file to see processing details."
                )
                
                upload_btn.click(
                    fn=interface.upload_and_process_pbix,
                    inputs=[file_input],
                    outputs=[upload_status, processing_summary],
                    show_progress=True
                )
            
            # Chat Tab
            with gr.Tab("Ask Questions"):
                gr.Markdown("### Ask Questions About Your Power BI Reports")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Question input
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., 'What measures are available in the Sales table?' or 'Explain the Total Revenue calculation'",
                            lines=2
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("Ask Question", variant="primary")
                            clear_btn = gr.Button("Clear Conversation", variant="secondary")
                        
                        # Settings
                        with gr.Accordion("Settings", open=False):
                            session_id = gr.Textbox(
                                label="Session ID",
                                value="default",
                                placeholder="Unique identifier for conversation tracking"
                            )
                            use_history = gr.Checkbox(
                                label="Use Conversation History",
                                value=True,
                                info="Include previous questions for context"
                            )
                            filter_type = gr.Dropdown(
                                label="Filter by Type",
                                choices=["All", "table", "measure", "column", "relationship", "visual", "page"],
                                value="All",
                                info="Filter results by artifact type"
                            )
                    
                    with gr.Column(scale=4):
                        # Answer display
                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            interactive=False
                        )
                        
                        # Context information
                        with gr.Accordion("Context & Sources", open=False):
                            context_output = gr.Textbox(
                                label="Retrieved Context",
                                lines=6,
                                interactive=False
                            )
                
                # Conversation history
                with gr.Accordion("Conversation History", open=False):
                    history_output = gr.Textbox(
                        label="Recent Questions & Answers",
                        lines=10,
                        interactive=False,
                        placeholder="Your conversation history will appear here..."
                    )
                
                # Event handlers
                ask_btn.click(
                    fn=interface.ask_question,
                    inputs=[question_input, session_id, use_history, filter_type],
                    outputs=[answer_output, context_output, history_output]
                )
                
                clear_btn.click(
                    fn=interface.clear_conversation,
                    inputs=[session_id],
                    outputs=[history_output]
                )
                
                # Enter key support
                question_input.submit(
                    fn=interface.ask_question,
                    inputs=[question_input, session_id, use_history, filter_type],
                    outputs=[answer_output, context_output, history_output]
                )
            
            # System Status Tab
            with gr.Tab("System Status"):
                gr.Markdown("### System Information and Status")
                
                status_display = gr.Markdown(
                    value=interface.get_system_status()
                )
                
                refresh_btn = gr.Button("Refresh Status")
                refresh_btn.click(
                    fn=interface.get_system_status,
                    outputs=[status_display]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Power BI RAG Assistant** - Built with Gradio, ChromaDB, OpenAI Embeddings, and Claude 3.5 Sonnet

        **Tips:**
        - Upload your PBIX files first to build the knowledge base
        - Ask specific questions about tables, measures, DAX formulas, or visualizations
        - Use conversation history for follow-up questions
        - Check system status if you encounter issues
        """)
    
    return app


def main():
    """Main function to launch the Gradio app."""
    
    app = create_app()
    
    app.launch(
        server_name=settings.ui.host,
        server_port=settings.ui.port,
        share=settings.ui.share,
        show_error=True,
        show_tips=True,
        enable_queue=True
    )


if __name__ == "__main__":
    main()
