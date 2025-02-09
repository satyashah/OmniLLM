import uvicorn
import logging
from serverRouter.router import app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)