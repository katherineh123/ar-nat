# flake8: noqa

# Import the generated workflow functions to trigger registration
from .ar_lab_assistant import (
    visual_process_guidance_function,
    rag_question_answering_function,
    experiment_logging_function
)
from .workflow import ar_lab_workflow
