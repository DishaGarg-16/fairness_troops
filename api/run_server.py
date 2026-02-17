import sys
from pathlib import Path

# Add project root to sys.path to ensure 'api' module can be imported
# This is necessary when running the script directly from a different directory
file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable, default to 8000
    # Railway injects the PORT variable automatically
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting server on port: {port}")
    
    # Run Uvicorn directly from Python
    uvicorn.run(
        "api.main:app", 
        host="0.0.0.0", 
        port=port,
        proxy_headers=True, # Important for running behind a proxy like Railway's
        forwarded_allow_ips="*"
    )
