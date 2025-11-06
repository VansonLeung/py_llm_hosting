import os
from dotenv import load_dotenv
import click
from src.libs.persistence import Persistence
from src.models.server import LLMServer
from src.libs.logging import logger

# Load environment variables from .env file
load_dotenv()

@click.group()
@click.option('--data-file', default='servers.json', help='Data file to use')
@click.pass_context
def cli(ctx, data_file):
    """LLM Endpoint Hosting CLI"""
    ctx.ensure_object(dict)
    ctx.obj['persistence'] = Persistence(data_file)

@cli.command()
@click.option('--name', required=True, help='Server name')
@click.option('--model', required=True, help='Model name')
@click.option('--mode', type=click.Choice(['proxy', 'self-hosted']), default='proxy', help='Server mode')
@click.option('--endpoint', help='Server endpoint URL (for proxy mode)')
@click.option('--model-path', help='HuggingFace model ID or local path (for self-hosted mode)')
@click.option('--backend', type=click.Choice(['llama-cpp', 'transformers', 'vllm', 'mlx', 'mlx-vlm', 'reranker', 'sentence-transformers']), help='Backend type (for self-hosted mode)')
@click.option('--gpu-layers', type=int, default=0, help='Number of GPU layers (llama-cpp)')
@click.option('--load-in-4bit', is_flag=True, help='Load in 4-bit (transformers)')
@click.option('--load-in-8bit', is_flag=True, help='Load in 8-bit (transformers)')
@click.option('--tensor-parallel', type=int, default=1, help='Tensor parallel size (vllm)')
@click.pass_context
def add_server(ctx, name, model, mode, endpoint, model_path, backend, gpu_layers, load_in_4bit, load_in_8bit, tensor_parallel):
    """Add a new LLM server"""
    persistence = ctx.obj['persistence']

    try:
        from src.models.server import ServerMode
        
        # Validate required fields based on mode
        if mode == 'proxy' and not endpoint:
            click.echo("Error: --endpoint is required for proxy mode", err=True)
            raise click.Abort()
        
        if mode == 'self-hosted':
            if not model_path:
                click.echo("Error: --model-path is required for self-hosted mode", err=True)
                raise click.Abort()
            if not backend:
                click.echo("Error: --backend is required for self-hosted mode", err=True)
                raise click.Abort()
        
        # Build backend config for self-hosted mode
        backend_config = {}
        if mode == 'self-hosted':
            if backend == 'llama-cpp':
                backend_config['n_gpu_layers'] = gpu_layers
            elif backend == 'transformers':
                backend_config['load_in_4bit'] = load_in_4bit
                backend_config['load_in_8bit'] = load_in_8bit
                backend_config['device'] = 'auto'
            elif backend == 'vllm':
                backend_config['tensor_parallel_size'] = tensor_parallel
            # mlx and mlx-vlm don't need special config
        
        server = LLMServer(
            name=name,
            model_name=model,
            mode=ServerMode(mode),
            endpoint_url=endpoint if mode == 'proxy' else None,
            model_path=model_path if mode == 'self-hosted' else None,
            backend_type=backend if mode == 'self-hosted' else None,
            backend_config=backend_config
        )
        
        persistence.add_server(server)
        logger.info(f"Server '{name}' added successfully")
        click.echo(f"Server '{name}' added successfully")
        
        if mode == 'self-hosted':
            click.echo(f"To load the model, start the server with: python main.py start")
    except Exception as e:
        logger.error(f"Failed to add server: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.pass_context
def list_servers(ctx):
    """List all servers"""
    persistence = ctx.obj['persistence']

    servers = persistence.get_servers()
    if not servers:
        click.echo("No servers configured")
        return

    for server in servers:
        click.echo(f"ID: {server.id}")
        click.echo(f"Name: {server.name}")
        click.echo(f"Model: {server.model_name}")
        click.echo(f"Mode: {server.mode.value}")
        if server.mode.value == 'proxy':
            click.echo(f"Endpoint: {server.endpoint_url}")
        else:
            click.echo(f"Model Path: {server.model_path}")
            click.echo(f"Backend: {server.backend_type}")
        click.echo(f"Status: {server.status.value}")
        click.echo("---")

@cli.command()
@click.option('--id', required=True, help='Server ID to remove')
@click.pass_context
def remove_server(ctx, id):
    """Remove a server"""
    persistence = ctx.obj['persistence']

    try:
        persistence.remove_server(id)
        logger.info(f"Server '{id}' removed successfully")
        click.echo(f"Server '{id}' removed successfully")
    except Exception as e:
        logger.error(f"Failed to remove server: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--port', default=lambda: int(os.environ.get('PORT', 8000)), help='Port to run the API server on')
@click.option('--host', default='0.0.0.0', help='Host to bind the API server to')
@click.pass_context
def start(ctx, port, host):
    """Start the API server"""
    import uvicorn
    from src.api import app
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option('--model-id', required=True, help='HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)')
@click.option('--filename', help='Specific file to download (for GGUF models)')
@click.option('--force', is_flag=True, help='Force re-download even if cached')
def download_model(model_id, filename, force):
    """Download a model from HuggingFace Hub"""
    try:
        from src.services.model_downloader import get_downloader
        
        click.echo(f"Downloading {model_id}...")
        if filename:
            click.echo(f"Specific file: {filename}")
        
        downloader = get_downloader()
        path = downloader.download_model(model_id, filename, force)
        
        click.echo(f"✓ Downloaded to: {path}")
        click.echo(f"\nYou can now use this path when adding a self-hosted server:")
        click.echo(f"  python main.py add-server --name <name> --model {model_id} \\")
        click.echo(f"    --mode self-hosted --model-path {path} --backend <backend>")
        
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nTo use model downloading, install: pip install huggingface-hub", err=True)
        raise click.Abort()
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
def list_loaded():
    """List currently loaded models"""
    try:
        import asyncio
        from src.services.model_manager import model_manager
        
        loaded = model_manager.list_loaded()
        
        if not loaded:
            click.echo("No models currently loaded")
            return
        
        click.echo("Loaded models:")
        for server_id, info in loaded.items():
            click.echo(f"\nServer ID: {server_id}")
            click.echo(f"  Backend: {info['backend_type']}")
            click.echo(f"  Capabilities: {', '.join(info['capabilities'])}")
    except Exception as e:
        logger.error(f"Failed to list loaded models: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--server-id', required=True, help='Server ID to unload')
def unload_model(server_id):
    """Unload a model and free resources"""
    try:
        import asyncio
        from src.services.model_manager import model_manager
        
        async def _unload():
            await model_manager.unload_model(server_id)
        
        asyncio.run(_unload())
        click.echo(f"✓ Model for server {server_id} unloaded")
    except KeyError:
        click.echo(f"Error: Server {server_id} not found or not loaded", err=True)
        raise click.Abort()
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()