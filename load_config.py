#!/usr/bin/env python3
"""
Load configuration from .env file
- Generates frontend config.js 
- Exports backend environment variables
"""

from pathlib import Path


def load_env_file(env_path: Path) -> dict:
    """Load variables from .env file."""
    env_vars = {}
    if not env_path.exists():
        return env_vars
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
    
    return env_vars


def generate_config_js(env_vars: dict, output_path: Path):
    """Generate config.js from environment variables."""
    backend_host = env_vars.get('BACKEND_HOST', 'localhost')
    backend_port = env_vars.get('BACKEND_PORT', '8000')
    
    config_content = f"""// AR Lab Assistant Frontend Configuration
// This file is AUTO-GENERATED from .env
// DO NOT EDIT MANUALLY - Run 'python3 load_config.py' to regenerate

const CONFIG = {{
    BACKEND_HOST: '{backend_host}',
    BACKEND_PORT: {backend_port},
    
    // Construct WebSocket URL
    get WEBSOCKET_URL() {{
        return `ws://${{this.BACKEND_HOST}}:${{this.BACKEND_PORT}}/websocket`;
    }}
}};
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)


def export_shell_vars(env_vars: dict):
    """Output shell export commands for backend environment variables."""
    backend_vars = ['NVIDIA_API_KEY', 'BACKEND_HOST', 'BACKEND_PORT', 'FRONTEND_PORT']
    
    for var in backend_vars:
        value = env_vars.get(var, '')
        if value:
            print(f'export {var}="{value}"')


def main():
    # Get project root (where .env is located)
    project_root = Path(__file__).parent
    env_path = project_root / '.env'
    config_output_path = project_root / 'ar_lab_assistant' / 'config.js'
    
    # Load environment variables from .env
    env_vars = load_env_file(env_path)
    
    # Generate config.js for frontend
    generate_config_js(env_vars, config_output_path)
    
    # Export environment variables for backend
    export_shell_vars(env_vars)


if __name__ == '__main__':
    main()
