"""OpenAI completions adapter for text-only evaluation of multimodal models."""

from transformers import AutoTokenizer

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import configure_pad_token


@register_model("local-text-completions")
class LocalTextCompletionsAPI(LocalCompletionsAPI):
    """Load only the text vocabulary from a multimodal model repository."""

    def __init__(
        self,
        tokenizer=None,
        tokenizer_backend="text",
        tokenized_requests=False,
        trust_remote_code=False,
        revision="main",
        use_fast_tokenizer=True,
        **kwargs,
    ):
        if tokenizer_backend != "text":
            raise ValueError("local-text-completions requires tokenizer_backend=text")

        tokenizer_name = tokenizer or kwargs.get("model") or kwargs.get("pretrained")
        if tokenizer_name is None:
            raise ValueError("local-text-completions requires a model or tokenizer")

        super().__init__(
            tokenizer=None,
            tokenizer_backend=None,
            tokenized_requests=tokenized_requests,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_fast_tokenizer=use_fast_tokenizer,
            **kwargs,
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_fast=use_fast_tokenizer,
            extra_special_tokens={},
        )
        self.tokenizer = configure_pad_token(text_tokenizer)
        self.tokenizer_backend = "huggingface"
        self.tokenized_requests = tokenized_requests
