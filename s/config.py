from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PROJECT_ROOT = PROJECT_ROOT.parent

AGENTIC_TOOLS = {
    "claude": SOURCE_PROJECT_ROOT / "claude",
    "copilot": SOURCE_PROJECT_ROOT / "copilot",
}
