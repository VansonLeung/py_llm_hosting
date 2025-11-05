#!/usr/bin/env python3
"""Main entry point for the LLM Hosting CLI."""
from src.cli.commands import cli

if __name__ == "__main__":
    # cli(obj={})
    import uvicorn
    from src.api import app
    uvicorn.run(app, host='0.0.0.0', port=8080)
