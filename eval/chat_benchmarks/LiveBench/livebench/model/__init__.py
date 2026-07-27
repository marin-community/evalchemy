def __getattr__(name):
    if name in {"load_model", "get_conversation_template", "add_model_args"}:
        from livebench.model import model_adapter

        return getattr(model_adapter, name)
    if name == "Model":
        from livebench.model.models import Model

        return Model
    if name == "get_model":
        from livebench.model.api_models import get_model

        return get_model
    raise AttributeError(name)
