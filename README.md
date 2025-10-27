# AR + NAT Demo

## Set-up Steps

Make a virtual environment
`cd ar_lab_assistant && uv venv --python 3.11 .venv`

Activate the virtual environment.
`source .venv/bin/activate`

Install dependencies
`uv pip install -e .`

Install RAG dependencies.
`pip install faiss-cpu sentence-transformers langchain-community langchain-text-splitters pypdf`


Add your NVIDIA API key.
`export NVIDIA_API_KEY="your-api-key-here"`

Start the nat backend server.
`nat serve --config_file configs/config.yml --port 8009 --host 0.0.0.0`
To see code changes, you can Ctrl+C to shut down the server and then restart it.

Start the CORS Proxy server (handles cross-origin requests between frontend and backend).
`python3 cors_proxy.py`

Start the Frontend HTTP Server.
`python3 -m http.server 8081`


Open the frontend in your browser by going to.
`http://[your-ip]:8081/simple_frontend.html`
