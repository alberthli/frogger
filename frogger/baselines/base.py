from dataclasses import dataclass, fields, make_dataclass
from typing import Callable

from frogger.robots.robot_core import RobotModelConfig


def combine_dataclasses(cls_a: type, cls_b: type) -> type:
    """Combines two config dataclasses into a new one, prioritizing attributes of cls_b.

    Parameters
    ----------
    cls_a : type
        The first dataclass type.
    cls_b : type
        The second dataclass type.

    Returns
    -------
    new_cls : type
        The new dataclass type.
    """
    # extract attributes from cls_a
    attributes = {
        field.name: (field.type, getattr(cls_a, field.name)) for field in fields(cls_a)
    }

    # remove fields that are methods in cls_b
    for k in list(attributes.keys()):
        if k in cls_b.__dict__ and isinstance(getattr(cls_b, k), Callable):
            del attributes[k]

    # include attributes from cls_b, prioritizing non-callable attributes
    for field in fields(cls_b):
        if field.name not in attributes and not isinstance(
            getattr(cls_b, field.name), Callable
        ):
            attributes[field.name] = field.type, getattr(cls_b, field.name)

    # combine methods, prioritizing those from B
    # note that we look through ALL attributes of cls_a, but only through the __dict__ of cls_b.
    # this is so that we have inherited methods but only override them with cls_b impls if they
    # are explicitly overridden.
    methods = {}
    for method_name in dir(cls_a):
        method = getattr(cls_a, method_name)
        if isinstance(method, Callable) and method_name not in cls_b.__dict__:
            methods[method_name] = method
    for method_name, method in cls_b.__dict__.items():
        if isinstance(method, Callable):
            methods[method_name] = method

    # generate the new class
    new_class_name = cls_a.__name__.replace("Config", "") + cls_b.__name__
    new_cls = make_dataclass(
        new_class_name,
        [(name, type_, default) for name, (type_, default) in attributes.items()],
        kw_only=True,
        bases=(cls_a, cls_b),
    )

    # add methods to the new dataclass
    for method_name, method in methods.items():
        if method_name != "__class__":
            setattr(new_cls, method_name, method)

    return new_cls


@dataclass(kw_only=True)
class BaselineConfig(RobotModelConfig):
    """Configuration for a generic baseline.

    Typically, we want to modify some RobotModelConfig that has already been
    instantiated with desirable properties. To do this, we will use the `from_cfg`
    method of BaselineConfig to load all of the attributes of an existing config
    into a baseline config that is the combination of the baseline attributes and
    the original config attributes.
    """

    @classmethod
    def from_cfg(cls, config_cls: type, **kwargs) -> "BaselineConfig":
        """Creates a baseline config from an existing robot model config class."""
        CombinedDataClass = combine_dataclasses(config_cls, cls)
        return CombinedDataClass
