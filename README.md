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
# Return to project root and load all configuration
cd .. && source <(python3 load_config.py -e)
```

This script:
- Reads `.env` and generates `ar_lab_assistant/config.js` for the frontend
- Exports all backend environment variables (`NVIDIA_API_KEY`, `BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_PORT`) to your current shell

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

Example with default values:
```
http://localhost:8080/websocket_frontend.html
```

## Quick Reference

### load_config.py Script

The `load_config.py` script provides a unified way to manage all configuration:

```bash
# Generate config.js and show export commands (informational)
python3 load_config.py

# Load environment variables into current shell
source <(python3 load_config.py -e)

# Only show export commands (no config.js generation)
python3 load_config.py --export-only

# Quiet mode (suppress informational messages)
python3 load_config.py -q
```

**What it does:**
- ✅ Reads `.env` file
- ✅ Generates `ar_lab_assistant/config.js` for frontend WebSocket URL
- ✅ Exports backend environment variables (`NVIDIA_API_KEY`, `BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_PORT`)
