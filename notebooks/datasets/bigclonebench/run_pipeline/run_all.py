import argparse
import subprocess
import sys
from pathlib import Path


STAGES = [
    ("00", "00_train_balanced_type123_type4_threshold.py"),
    ("01", "01_train_type123_threshold_test_type4.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCB cross-type threshold experiments.")
    parser.add_argument("--start-at", choices=[stage for stage, _ in STAGES], default="00")
    parser.add_argument("--stop-after", choices=[stage for stage, _ in STAGES], default="01")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    selected = [
        (stage, script)
        for stage, script in STAGES
        if int(args.start_at) <= int(stage) <= int(args.stop_after)
    ]
    if not selected:
        raise ValueError("--start-at must be earlier than or equal to --stop-after.")

    for stage, script in selected:
        script_path = run_dir / script
        print(f"\n[*] Running stage {stage}: {script_path}")
        subprocess.run([sys.executable, str(script_path)], cwd=str(run_dir), check=True)

    print("\n[+] All selected BCB cross-type pipeline stages completed.")


if __name__ == "__main__":
    main()
