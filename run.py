"""
GlucoGuard: Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
Unified Project Runner (run.py)
IBM SkillsBuild / Bharat Cares AICTE Machine Learning Final Submission

Supported Modes:
  - python run.py          : Boots Streamlit Clinical Dashboard (Default: http://localhost:8501)
  - python run.py --api    : Boots FastAPI REST Diagnostic Microservice (http://127.0.0.1:8000)
  - python run.py --all    : Concurrently spawns both Streamlit UI and FastAPI Microservice
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_diabetes_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
APP_PATH = PROJECT_ROOT / "app.py"
API_PATH = PROJECT_ROOT / "api.py"
TRAINER_PATH = PROJECT_ROOT / "src" / "model_trainer.py"

BANNER = r"""
  ============================================================================
    ____ _                    ____                     _ 
   / ___| |_   _  ___ ___    / ___|_   _  __ _ _ __ __| |
  | |  _| | | | |/ __/ _ \  | |  _| | | |/ _` | '__/ _` |
  | |_| | | |_| | (_| (_) | | |_| | |_| | (_| | | | (_| |
   \____|_|\__,_|\___\___/   \____|\__,_|\__,_|_|  \__,_|
   
   Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
   IBM SkillsBuild | Bharat Cares | AICTE ML Internship Final Submission
  ============================================================================
"""


def verify_environment(mode: str = "ui"):
    """Validates that required core Python dependencies are present."""
    base_required = ["pandas", "numpy", "sklearn", "joblib"]
    ui_required = ["streamlit", "plotly", "fpdf"]
    api_required = ["fastapi", "uvicorn", "pydantic"]

    needed = base_required
    if mode in ["ui", "all"]:
        needed += ui_required
    if mode in ["api", "all"]:
        needed += api_required

    missing = []
    for pkg in set(needed):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[!] Warning: Missing recommended packages: {missing}")
        print(f"[!] Run: pip install -r requirements.txt\n")


def check_and_train_models():
    """Verifies existing model artifacts, auto-training if missing."""
    if not BEST_MODEL_PATH.exists() or not METADATA_PATH.exists():
        print("[*] Pre-trained models not found. Initiating automated training sequence...")
        print("[*] Running: src/model_trainer.py")
        cmd = [sys.executable, str(TRAINER_PATH)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[X] Error occurred during model training. Exiting.")
            sys.exit(result.returncode)
        print("[✓] Model training and cross-validation completed successfully!\n")
    else:
        print("[✓] Serialized clinical models & benchmark metadata verified.\n")


def launch_streamlit_app():
    """Launches the Streamlit clinical dashboard application (blocking)."""
    print("[*] Starting GlucoGuard Clinical Dashboard at http://localhost:8501 ...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n[*] Shutting down Streamlit application. Goodbye!")


def launch_fastapi_server():
    """Launches the FastAPI REST Diagnostic Microservice via Uvicorn (blocking)."""
    print("[*] Starting GlucoGuard FastAPI Microservice at http://127.0.0.1:8000 ...")
    print("[*] Interactive API Documentation available at http://127.0.0.1:8000/docs")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload"
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n[*] Shutting down FastAPI microservice. Goodbye!")


def launch_all_concurrent():
    """Spawns both Streamlit and FastAPI microservices concurrently."""
    print("[*] Initiating Dual-Delivery Mode (FastAPI Microservice + Streamlit Dashboard)...")

    # Launch FastAPI Microservice
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]
    api_proc = subprocess.Popen(api_cmd, cwd=str(PROJECT_ROOT))
    print("[✓] FastAPI Microservice running in background at http://127.0.0.1:8000 (Docs: /docs)")

    # Small pause to allow API initialization
    time.sleep(1.5)

    # Launch Streamlit
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    print("[✓] Launching Streamlit Clinical Dashboard at http://localhost:8501 ...")
    ui_proc = subprocess.Popen(ui_cmd, cwd=str(PROJECT_ROOT))

    try:
        # Keep main thread alive waiting for processes
        while True:
            time.sleep(1)
            if api_proc.poll() is not None or ui_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n[*] Shutting down all GlucoGuard services cleanly...")
    finally:
        if api_proc.poll() is None:
            api_proc.terminate()
        if ui_proc.poll() is None:
            ui_proc.terminate()
        print("[✓] All services terminated safely.")


def main():
    parser = argparse.ArgumentParser(description="GlucoGuard Clinical AI Engine Runner")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch the FastAPI REST Diagnostic Microservice only (http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Launch both Streamlit Dashboard and FastAPI Microservice concurrently"
    )
    args = parser.parse_args()

    print(BANNER)

    mode = "all" if args.all else ("api" if args.api else "ui")
    verify_environment(mode=mode)
    check_and_train_models()

    if args.all:
        launch_all_concurrent()
    elif args.api:
        launch_fastapi_server()
    else:
        launch_streamlit_app()


if __name__ == "__main__":
    main()
