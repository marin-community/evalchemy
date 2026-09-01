from types import SimpleNamespace

from eval.lm_eval_models import text_completions


class FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def batch_decode(self, token_batches):
        return ["".join(chr(token) for token in tokens) for tokens in token_batches]


def test_text_completions_tokenizes_rolling_likelihood_prompts(monkeypatch):
    def load_text_tokenizer(*args, extra_special_tokens=None, **kwargs):
        del args, kwargs
        if extra_special_tokens != {}:
            raise AttributeError("multimodal extra_special_tokens must be disabled")
        return FakeTokenizer()

    monkeypatch.setattr(
        text_completions.AutoTokenizer,
        "from_pretrained",
        load_text_tokenizer,
    )

    model = text_completions.LocalTextCompletionsAPI(
        model="multimodal-base-model",
        base_url="http://localhost:8000/v1/completions",
        tokenizer_backend="text",
        tokenized_requests=False,
    )

    assert model.tok_encode("abc") == [97, 98, 99]
    assert model.create_message([[97, 98, 99]]) == "abc"
