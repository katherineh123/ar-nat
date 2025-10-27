# AR + NAT Demo

## Configuration

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

Update it with your values if desired.


## Set-up Steps

### 1. Make a virtual environment
```bash
cd ar_lab_assistant && uv venv --python 3.11 .venv
```

### 2. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -e .
```

### 4. Install RAG dependencies
```bash
pip install faiss-cpu sentence-transformers langchain-community langchain-text-splitters pypdf
```

### 5. Load configuration from .env
```bash
# Return to project root and load configuration
cd .. && source <(python3 load_config.py)
```

This generates `ar_lab_assistant/config.js` for the frontend and exports backend environment variables.

### 6. Start the NAT backend server
```bash
cd ar_lab_assistant && nat serve --config_file src/ar_lab_assistant/configs/config.yml --port ${BACKEND_PORT:-8000} --host 0.0.0.0
```

To see code changes, you can Ctrl+C to shut down the server and then restart it.

### 7. Start the Frontend HTTP Server (in another terminal)
```bash
cd ar_lab_assistant && python3 -m http.server ${FRONTEND_PORT:-8080}
```

### 8. Open the frontend in your browser
```
http://${BACKEND_HOST}:${FRONTEND_PORT}/websocket_frontend.html
```

For example, it might look something like this (using default values here):
```
http://localhost:8080/websocket_frontend.html
```

## Example Inputs

### Example Questions
- "What tools are needed for this experiment?"
- "Why are forceps needed for this experiment?"

### Trigger VPG Workflow
- "Let's get started"

### Log Session / End Session
- "Can you log this session?"


## WIP Features
- Reduce latency of the ReAct agent in the Q&A function
- Increase overall robustness (add better conversation history, Internet Search tool fallback, etc)
- Add a VLM tool to the ReAct agent in the Q&A function
- Display logged past conversations in the UI
- Code clean-up
