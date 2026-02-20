"""Launch both the API server and Streamlit UI with a single command."""
import subprocess
import sys
import time

def main():
    python = sys.executable

    # Start API server in background
    print("Starting API server...")
    api = subprocess.Popen([python, "-m", "uvicorn", "api.main:app", "--reload"])
    time.sleep(3)

    # Start Streamlit UI
    print("Starting Streamlit UI...")
    ui = subprocess.Popen([python, "-m", "streamlit", "run", "ui/app.py"])

    try:
        api.wait()
        ui.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api.terminate()
        ui.terminate()

if __name__ == "__main__":
    main()
