"""Load Gemma 4 tokenizers with the Transformers 4 client used by lm-eval."""

from transformers import AutoTokenizer
from transformers.models.auto.tokenization_auto import get_tokenizer_config

GEMMA4_VIDEO_TOKEN = "<|video|>"
_original_from_pretrained = AutoTokenizer.from_pretrained


def _from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    try:
        return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    except AttributeError as exc:
        if str(exc) != "'list' object has no attribute 'keys'" or "extra_special_tokens" in kwargs:
            raise

        config_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "local_files_only", "subfolder")
            if key in kwargs
        }
        config = get_tokenizer_config(pretrained_model_name_or_path, **config_kwargs)
        if config.get("extra_special_tokens") != [GEMMA4_VIDEO_TOKEN]:
            raise

        return _original_from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            extra_special_tokens={"video_token": GEMMA4_VIDEO_TOKEN},
        )


def install_tokenizer_compat():
    """Patch tokenizer loads process-wide for Gemma 4's Transformers 5 config."""
    if getattr(AutoTokenizer.from_pretrained, "__func__", None) is _from_pretrained:
        return
    AutoTokenizer.from_pretrained = classmethod(_from_pretrained)
