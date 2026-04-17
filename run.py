"""Convenience launcher – runs the Streamlit app."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "main.py", "--server.port", "8501"],
        check=True,
    )
