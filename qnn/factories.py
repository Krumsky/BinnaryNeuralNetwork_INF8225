import qnn.models as mod
import qnn.freeze as freeze

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
    
def freeze_agent_factory(freeze_agent_name: str, kwargs: dict) -> freeze.FreezeAgent:
    """
    Factory function to create a freeze agent based on the provided name and arguments.

    Args:
        freeze_agent_name (str): The name of the freeze agent to create.
        args (list): A list of arguments to pass to the freeze agent's constructor.

    Returns:
        mod.FreezeAgent: An instance of the specified freeze agent.
    """
    if hasattr(freeze, freeze_agent_name):
        freeze_agent_class = getattr(freeze, freeze_agent_name)
        return freeze_agent_class(**kwargs)
    else:
        raise ValueError(f"Freeze agent {freeze_agent_name} not found in qnn.freeze.")
