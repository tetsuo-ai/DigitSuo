#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000
DOCS_DIR = Path("docs/html")

def main():
    if not DOCS_DIR.exists():
        print("Error: Documentation not found!")
        print("Please run 'doxygen' first to generate the documentation.")
        return

    os.chdir(DOCS_DIR)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"\nServing documentation at http://0.0.0.0:{PORT}")
        print("(Use this URL if accessing from another machine)")
        print(f"\nLocal access: http://localhost:{PORT}")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            webbrowser.open(f"http://localhost:{PORT}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()
            httpd.server_close()
            print("Server stopped")

if __name__ == "__main__":
    main()
