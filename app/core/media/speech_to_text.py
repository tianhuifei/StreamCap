import os
import subprocess
import tempfile
from functools import lru_cache

from ... import execute_dir
from ...utils.logger import logger

WHISPER_MODELS_DIR = os.path.join(execute_dir, "models", "whisper")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".wma", ".flac", ".ogg"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | {".mp4", ".ts", ".mkv", ".mov", ".flv", ".nut"}


def collect_recording_output_files(save_file_path: str, prefer_mp4: bool = False) -> list[str]:
    """Collect recorded media files from a save path or segmented template path."""
    save_file_path = save_file_path.replace("\\", "/")
    directory = os.path.dirname(save_file_path)
    basename = os.path.basename(save_file_path)

    if not os.path.isdir(directory):
        return []

    if "%03d" in basename:
        prefix = basename.split("_%03d", maxsplit=1)[0]
        preferred_ext = ".mp4" if prefer_mp4 else os.path.splitext(basename)[1].lower()
        files = []
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if not name.startswith(f"{prefix}_") or ext not in MEDIA_EXTENSIONS:
                continue
            if prefer_mp4 and ext != ".mp4":
                continue
            if not prefer_mp4 and ext != preferred_ext:
                continue
            files.append(path.replace("\\", "/"))
        return sorted(files)

    if prefer_mp4 and save_file_path.lower().endswith(".ts"):
        mp4_path = save_file_path.rsplit(".", maxsplit=1)[0] + ".mp4"
        if os.path.exists(mp4_path):
            return [mp4_path.replace("\\", "/")]

    if os.path.exists(save_file_path):
        return [save_file_path]

    return []


def _is_audio_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in AUDIO_EXTENSIONS


def _extract_audio(source_path: str, output_path: str) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore").strip()
            logger.error(f"Failed to extract audio from {source_path}: {stderr}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except OSError as e:
        logger.error(f"Failed to extract audio from {source_path}: {e}")
        return False


@lru_cache(maxsize=1)
def _get_opencc_converter():
    try:
        from opencc import OpenCC
    except ImportError as e:
        raise ImportError("opencc-python-reimplemented is not installed. Install it with: pip install opencc-python-reimplemented") from e

    return OpenCC("t2s")


def to_simplified_chinese(text: str) -> str:
    try:
        return _get_opencc_converter().convert(text)
    except ImportError as e:
        logger.warning(str(e))
        return text
    except Exception as e:
        logger.warning(f"Failed to convert text to simplified Chinese: {e}")
        return text


def _get_whisper_device_config() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as e:
        logger.debug(f"CUDA not available for speech-to-text: {e}")
    return "cpu", "int8"


def _get_local_whisper_model_path(model_name: str) -> str:
    local_model_dir = os.path.join(WHISPER_MODELS_DIR, model_name)
    if not os.path.isfile(os.path.join(local_model_dir, "model.bin")):
        raise FileNotFoundError(
            f"Local speech-to-text model not found: {local_model_dir}. "
            f"Download it with: python app/scripts/download_whisper_model.py {model_name}"
        )
    return local_model_dir


@lru_cache(maxsize=4)
def _get_whisper_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper is not installed. Install it with: pip install faster-whisper"
        ) from e

    device, compute_type = _get_whisper_device_config()
    model_path = _get_local_whisper_model_path(model_name)
    logger.info(
        f"Loading speech-to-text model: {model_path} (device={device}, compute_type={compute_type})"
    )
    return WhisperModel(model_path, device=device, compute_type=compute_type)


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_segment_text(segments) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        text = to_simplified_chinese(text)
        lines.append(f"[{_format_timestamp(segment.start)}] {text}")
    return "\n\n".join(lines)


def transcribe_media_file(
    media_path: str,
    model_name: str = "medium",
    language: str | None = None,
) -> str | None:
    """Transcribe an audio or video file and return timestamped plain text."""
    media_path = media_path.replace("\\", "/")
    if not os.path.exists(media_path) or os.path.getsize(media_path) == 0:
        logger.warning(f"Skip speech-to-text, file missing or empty: {media_path}")
        return None

    temp_audio_path = None
    input_path = media_path

    try:
        if not _is_audio_file(media_path):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_file.name
            temp_file.close()
            if not _extract_audio(media_path, temp_audio_path):
                return None
            input_path = temp_audio_path

        model = _get_whisper_model(model_name)
        segments, _info = model.transcribe(
            input_path,
            language=language or "zh",
            vad_filter=True,
        )
        text = _format_segment_text(segments)
        return text or None
    except ImportError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Speech-to-text failed for {media_path}: {e}")
        return None
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def save_transcript(media_path: str, text: str) -> str:
    output_path = media_path.rsplit(".", maxsplit=1)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)
    return output_path


def transcribe_and_save(
    media_path: str,
    model_name: str = "medium",
    language: str | None = None,
) -> str | None:
    text = transcribe_media_file(media_path, model_name=model_name, language=language)
    if not text:
        return None
    output_path = save_transcript(media_path, text)
    logger.info(f"Speech-to-text completed: {output_path}")
    return output_path
