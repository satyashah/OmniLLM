import uvicorn
import logging
from serverRouter.router import app
import os #added to get env variable

# Configure logging, and the logging file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="server.log")  # Add filename to log to a file


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.exception("Exception during server startup:")

print("server.py executed") #print to show server has executed