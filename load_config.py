#!/usr/bin/env python3
"""
Load configuration from .env file
- Generates frontend config.js 
- Exports backend environment variables
This ensures frontend and backend configurations stay in sync.
"""

import argparse
import os
import sys
from pathlib import Path


def load_env_file(env_path: Path) -> dict:
    """Load variables from .env file."""
    env_vars = {}
    if not env_path.exists():
        print(f"Warning: {env_path} not found. Using defaults.", file=sys.stderr)
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


def generate_config_js(env_vars: dict, output_path: Path, verbose: bool = True):
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
    
    if verbose:
        print(f"✅ Generated {output_path}", file=sys.stderr)
        print(f"   Backend: {backend_host}:{backend_port}", file=sys.stderr)


def export_shell_vars(env_vars: dict, verbose: bool = True):
    """Output shell export commands for backend environment variables."""
    # List of variables needed for backend
    backend_vars = ['NVIDIA_API_KEY', 'BACKEND_HOST', 'BACKEND_PORT', 'FRONTEND_PORT']
    
    if verbose:
        print("# Export these environment variables for backend:", file=sys.stderr)
    
    for var in backend_vars:
        value = env_vars.get(var, '')
        if value:
            # Output export command to stdout (can be sourced)
            print(f'export {var}="{value}"')
        elif verbose:
            print(f"# Warning: {var} not found in .env", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Load configuration from .env file',
        epilog='Examples:\n'
               '  python3 load_config.py              # Generate config.js and show exports\n'
               '  python3 load_config.py --export     # Only output export commands\n'
               '  source <(python3 load_config.py -e) # Source exports into current shell\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-e', '--export-only',
        action='store_true',
        help='Only output shell export commands (no config.js generation)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress informational messages'
    )
    
    args = parser.parse_args()
    
    # Get project root (where .env is located)
    project_root = Path(__file__).parent
    env_path = project_root / '.env'
    config_output_path = project_root / 'ar_lab_assistant' / 'config.js'
    
    # Load environment variables
    env_vars = load_env_file(env_path)
    
    verbose = not args.quiet
    
    if args.export_only:
        # Only output export commands (useful for sourcing)
        export_shell_vars(env_vars, verbose=False)
    else:
        # Default: Generate config.js and output export commands
        generate_config_js(env_vars, config_output_path, verbose=verbose)
        
        if verbose:
            print("\n" + "="*60, file=sys.stderr)
        
        export_shell_vars(env_vars, verbose=verbose)
        
        if verbose:
            print("="*60, file=sys.stderr)
            print("\n💡 To load these variables in your current shell, run:", file=sys.stderr)
            print("   source <(python3 load_config.py -e)", file=sys.stderr)


if __name__ == '__main__':
    main()
