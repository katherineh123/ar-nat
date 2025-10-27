"""
Log Session Node for AR Lab Assistant workflow.
Logs the session to a database and ends the workflow.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class LogSessionNodeConfig(FunctionBaseConfig, name="log_session_node"):
    """Log session node configuration."""
    db_path: str = Field(
        default="data/experiment_logs.db",
        description="Path to the SQLite database file"
    )
    experiment_type: str = Field(
        default="Kirby-Bauer disk diffusion assay",
        description="Type of experiment being performed"
    )


@register_function(config_type=LogSessionNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def log_session_node_function(config: LogSessionNodeConfig, builder: Builder):
    """Log Session node - logs the session and ends."""
    
    async def _log_session_node(state: dict) -> dict:
        """Log Session node - logs the session and ends."""
        # Create conversation summary
        conversation_summary = "AR Lab Assistant session completed successfully. Student performed Kirby-Bauer disk diffusion assay experiment."
        
        logger.info("Logging experiment session")
        
        # Ensure data directory exists
        data_dir = Path(config.db_path).parent
        data_dir.mkdir(exist_ok=True)
        
        # Initialize database if it doesn't exist
        conn = sqlite3.connect(config.db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scientist_name TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                conversation_summary TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Get the next student number
        cursor.execute('SELECT COUNT(*) FROM experiment_logs')
        student_count = cursor.fetchone()[0]
        scientist_name = f"Student {student_count + 1}"
        
        # Insert the log entry
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO experiment_logs (timestamp, scientist_name, experiment_type, conversation_summary)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, scientist_name, config.experiment_type, conversation_summary))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Experiment logged for {scientist_name}")
        result = f"Experiment session logged successfully for {scientist_name}. Session ended."
        
        response = f"{result}\n\nThank you for using the AR Lab Assistant!"
        
        state["messages"].append(AIMessage(content=response))
        state["current_node"] = "log_session"
        state["session_data"]["session_ended"] = True
        
        logger.info("Log Session: Session logged and ended")
        
        return state
    
    yield FunctionInfo.from_fn(_log_session_node, description="Log Session node - logs session and ends")
