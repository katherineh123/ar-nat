#!/usr/bin/env python3
"""
Simple CORS proxy server to handle requests from the frontend to the NAT API.
This solves the CORS issue when the frontend and API are on different ports.
"""

import asyncio
import aiohttp
from aiohttp import web
import json

# Configuration
NAT_API_URL = "http://localhost:8009"
PROXY_PORT = 8008

async def handle_chat(request):
    """Proxy chat requests to the NAT API with CORS headers."""
    try:
        # Get the request data
        data = await request.json()
        
        # Forward the request to the NAT API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{NAT_API_URL}/chat",
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_data = await response.json()
                
                # Return response with CORS headers
                return web.json_response(
                    response_data,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type",
                    }
                )
                
    except Exception as e:
        print(f"Error proxying request: {e}")
        return web.json_response(
            {"error": "Proxy error"},
            status=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

async def handle_options(request):
    """Handle CORS preflight requests."""
    return web.Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

def create_app():
    """Create the web application."""
    app = web.Application()
    
    # Add routes
    app.router.add_post("/chat", handle_chat)
    app.router.add_options("/chat", handle_options)
    
    return app

async def main():
    """Start the proxy server."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "0.0.0.0", PROXY_PORT)
    await site.start()
    
    print(f"CORS proxy server running on http://0.0.0.0:{PROXY_PORT}")
    print(f"Proxying requests to NAT API at {NAT_API_URL}")
    
    # Keep the server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("Shutting down proxy server...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
