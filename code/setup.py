# setup.py
from setuptools import setup, find_packages

setup(
    name="journal_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai-whisper",
        "ollama",
        "chromadb",
        "sentence-transformers",
    ],
    entry_points={
        "console_scripts": [
            "journal=journal_assistant:main",
        ],
    },
)
