"""Merge generation intent using the caller's explicit settings last."""


def generation_parameters(defaults, overrides):
    parameters = {**defaults, **overrides}
    if "temperature" in overrides and "do_sample" not in overrides:
        parameters["do_sample"] = float(overrides["temperature"]) > 0
    return parameters
