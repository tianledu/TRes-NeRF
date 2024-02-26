models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name):
    model = models[name]#(config)
    return model

# from . import dysdf, fields, deformation_nets
from . import fields
