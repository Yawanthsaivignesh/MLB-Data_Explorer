# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Retrieve API key securely
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
print(f"API Key: {GEMINI_API_KEY}")

# Optional: Add validation
if not GEMINI_API_KEY:
    raise ValueError("No API key found. Please set GEMINI_API_KEY in .env file.")