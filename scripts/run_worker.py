"""Standalone Autonomous Investigation Worker Daemon Entrypoint."""
import argparse
import os
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from packages.analytics_core.src.execution.worker import WorkerSupervisor


def main():
    parser = argparse.ArgumentParser(description="AA-OS Autonomous Investigation Worker Daemon")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker node identifier")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Queue polling interval in seconds")
    args = parser.parse_args()

    supervisor = WorkerSupervisor(
        worker_id=args.worker_id,
        poll_interval_seconds=args.poll_interval,
    )

    def handle_signal(sig, frame):
        print(f"\n[Worker] Received shutdown signal ({sig}). Stopping supervisor...")
        supervisor.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    supervisor.run_loop()


if __name__ == "__main__":
    main()
