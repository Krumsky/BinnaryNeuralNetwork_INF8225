import qnn.models as mod

def model_factory(model_name: str, kwargs: dict) -> mod.ModelBase:
    """
    Factory function to create a model based on the provided name and arguments.

    Args:
        model_name (str): The name of the model to create.
        args (list): A list of arguments to pass to the model's constructor.

    Returns:
        mod.ModelBase: An instance of the specified model.
    """
    if hasattr(mod, model_name):
        model_class = getattr(mod, model_name)
        return model_class(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not found in qnn.models.")
