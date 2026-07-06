import logging
import os
import shutil
import sys
import threading
from pathlib import Path

APP_NAME = "StreamCap"
EXECUTABLE_SUFFIX = ".exe" if sys.platform == "win32" else ""
BUNDLED_WHISPER_MODELS = ("medium",)  # keep in sync with scripts/build.py
WHISPER_MODEL_COPY_TIMEOUT_SECONDS = 600.0

logger = logging.getLogger(__name__)

_whisper_copy_state_lock = threading.Lock()
_whisper_copy_thread: threading.Thread | None = None
_whisper_copy_complete = threading.Event()
_whisper_copy_failed: BaseException | None = None


def _executable_dir() -> Path:
    executable = sys.executable if getattr(sys, "frozen", False) else sys.argv[0]
    return Path(executable).resolve().parent


def _macos_bundle_contents_dir(executable_dir: Path) -> Path | None:
    contents_dir = executable_dir.parent
    app_dir = contents_dir.parent
    if (
        getattr(sys, "frozen", False)
        and sys.platform == "darwin"
        and executable_dir.name == "MacOS"
        and contents_dir.name == "Contents"
        and app_dir.suffix == ".app"
    ):
        return contents_dir
    return None


_EXECUTABLE_DIR = _executable_dir()
_CONTENTS_DIR = _macos_bundle_contents_dir(_EXECUTABLE_DIR)

if _CONTENTS_DIR is not None:
    resource_dir = _CONTENTS_DIR / "Resources"
    user_data_dir = Path.home() / "Library" / "Application Support" / APP_NAME
elif getattr(sys, "frozen", False) and sys.platform == "win32":
    internal_dir = _EXECUTABLE_DIR / "_internal"
    resource_dir = internal_dir if internal_dir.is_dir() else _EXECUTABLE_DIR
    user_data_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
else:
    resource_dir = _EXECUTABLE_DIR
    user_data_dir = _EXECUTABLE_DIR

if getattr(sys, "frozen", False) and sys.platform == "win32":
    default_recordings_dir = _EXECUTABLE_DIR / "downloads"
else:
    default_recordings_dir = user_data_dir / "downloads"


def prepare_user_data_dir() -> None:
    """Copy bundled defaults to the writable user data directory when needed."""
    user_data_dir.mkdir(parents=True, exist_ok=True)
    if user_data_dir == resource_dir:
        return

    for directory in ("config", "locales"):
        source = resource_dir / directory
        target = user_data_dir / directory
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)

    prepare_bundled_ffmpeg()
    prepare_bundled_node()
    start_bundled_whisper_model_copy()


def prepare_bundled_ffmpeg() -> None:
    source_executable = resource_dir / "ffmpeg" / f"ffmpeg{EXECUTABLE_SUFFIX}"
    target_dir = user_data_dir / "ffmpeg"
    target_executable = target_dir / f"ffmpeg{EXECUTABLE_SUFFIX}"
    if not source_executable.is_file() or target_executable.exists():
        return
    if shutil.which("ffmpeg"):
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_executable, target_executable)
    if sys.platform != "win32":
        target_executable.chmod(target_executable.stat().st_mode | 0o755)


def prepare_bundled_node() -> None:
    source_executable = resource_dir / "node" / f"node{EXECUTABLE_SUFFIX}"
    target_dir = user_data_dir / "node"
    target_executable = target_dir / f"node{EXECUTABLE_SUFFIX}"
    if not source_executable.is_file() or target_executable.exists():
        return
    if shutil.which("node"):
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_executable, target_executable)
    if sys.platform != "win32":
        target_executable.chmod(target_executable.stat().st_mode | 0o755)


def whisper_model_target_dir(model_name: str) -> Path:
    return user_data_dir / "models" / "whisper" / model_name


def whisper_model_is_ready(model_name: str) -> bool:
    return (whisper_model_target_dir(model_name) / "model.bin").is_file()


def _whisper_models_need_copy() -> bool:
    for model_name in BUNDLED_WHISPER_MODELS:
        source_model = resource_dir / "models" / "whisper" / model_name / "model.bin"
        if source_model.is_file() and not whisper_model_is_ready(model_name):
            return True
    return False


def _copy_bundled_whisper_models() -> None:
    for model_name in BUNDLED_WHISPER_MODELS:
        source_dir = resource_dir / "models" / "whisper" / model_name
        target_dir = whisper_model_target_dir(model_name)
        source_model = source_dir / "model.bin"
        target_model = target_dir / "model.bin"
        if not source_model.is_file() or target_model.is_file():
            continue
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir)


def _run_whisper_model_copy() -> None:
    global _whisper_copy_failed
    try:
        logger.info("Copying bundled Whisper models to user data directory in background...")
        _copy_bundled_whisper_models()
        logger.info("Bundled Whisper models are ready.")
    except BaseException as exc:
        _whisper_copy_failed = exc
        logger.error("Failed to copy bundled Whisper models: %s", exc)
    finally:
        _whisper_copy_complete.set()


def start_bundled_whisper_model_copy() -> None:
    """Start copying bundled Whisper models in a background thread when needed."""
    global _whisper_copy_thread, _whisper_copy_failed

    if user_data_dir == resource_dir:
        _whisper_copy_complete.set()
        return

    if not _whisper_models_need_copy():
        _whisper_copy_complete.set()
        return

    with _whisper_copy_state_lock:
        if _whisper_copy_thread is not None and _whisper_copy_thread.is_alive():
            return

        _whisper_copy_complete.clear()
        _whisper_copy_failed = None
        _whisper_copy_thread = threading.Thread(
            target=_run_whisper_model_copy,
            name="BundledWhisperModelCopy",
            daemon=True,
        )
        _whisper_copy_thread.start()


def ensure_whisper_model_ready(model_name: str, timeout: float = WHISPER_MODEL_COPY_TIMEOUT_SECONDS) -> Path:
    """Wait until the requested Whisper model is available in the user data directory."""
    target_dir = whisper_model_target_dir(model_name)
    if whisper_model_is_ready(model_name):
        return target_dir

    start_bundled_whisper_model_copy()
    if not _whisper_copy_complete.wait(timeout=timeout):
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s while preparing speech-to-text model: {model_name}"
        )
    if _whisper_copy_failed is not None:
        raise RuntimeError(f"Failed to prepare speech-to-text model: {model_name}") from _whisper_copy_failed
    if not whisper_model_is_ready(model_name):
        raise FileNotFoundError(
            f"Speech-to-text model not found: {target_dir}. "
            f"Download it with: python app/scripts/download_whisper_model.py {model_name}"
        )
    return target_dir


def prepend_user_bin_dirs() -> None:
    for directory in (user_data_dir / "ffmpeg", user_data_dir / "node"):
        if directory.is_dir():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
