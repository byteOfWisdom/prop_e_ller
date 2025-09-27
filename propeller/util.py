def list_like(var):
    return hasattr(var, "__getitem__") and hasattr(var, "__len__")
