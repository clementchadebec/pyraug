import typing

def load_model_from_string(model_name: str):
    """Utils method to load a model from its name in string"""
    try:
        mod = __import__('pyraug.models', fromlist=[model_name])
        model_class = getattr(mod, model_name)
    
    except:
        raise ModuleNotFoundError(f"Enable to locate model: '{model_name}' in "
            "pyraug.models. Maybe not implemented. Check documentation.")

    return model_class
