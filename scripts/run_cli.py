try:
    from contract_copilot.cli import run_cli_main
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_ROOT = PROJECT_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from contract_copilot.cli import run_cli_main


if __name__ == "__main__":
    raise SystemExit(run_cli_main())
