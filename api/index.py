"""Vercel serverless entrypoint for the Model Router."""

import sys
import os

# Add parent directory to path so model_router package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_router.server import app  # noqa: E402
