"""App package exposing the FastAPI instance.

Usage: uvicorn app:app --reload
"""

from localinferenceapi import app as app  # re-export unified predictor backend
