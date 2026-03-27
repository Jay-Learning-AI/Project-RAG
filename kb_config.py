from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def load_settings() -> None:
    """Load local .env for development without overriding environment-provided secrets."""
    if load_dotenv is None:
        return

    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)