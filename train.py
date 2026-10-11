"""Convenience entry point; install package or use the repository src directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tool_rl.train import main

if __name__ == "__main__":
    main()

