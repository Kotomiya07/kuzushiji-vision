"""Utility classes and functions."""

from typing import Any


class EasyDict(dict):
    """A dictionary subclass that allows attribute-style access to dictionary items,
    including nested dictionaries.
    This class extends the built-in dict class to enable accessing dictionary keys
    as if they were object attributes, providing a more intuitive interface.
    Nested dictionaries are also converted to EasyDict instances recursively.
    Example:
        >>> d = EasyDict({'foo': 1, 'bar': {'baz': 2}})
        >>> d.foo
        1
        >>> d.bar.baz
        2
        >>> d.bar.qux = 3
        >>> d['bar']['qux']
        3
    Attributes inherit from dict, plus:
        Any key-value pairs added to the dictionary can be accessed as attributes.
    Raises:
        AttributeError: When trying to access a non-existent key as an attribute.
    """

    def __init__(self, d: dict = None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = EasyDict(value)
        super().__setattr__(name, value)
        # Ensure the item is also set in the dictionary view
        super().__setitem__(name, value)

    # __setitem__ is called by dict methods like update
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = EasyDict(value)
        super().__setitem__(key, value)
        # Ensure the attribute view is also updated
        super().__setattr__(key, value)

    def __getattr__(self, name: str) -> Any:
        try:
            # Attempt to get value from dict first
            return self[name]
        except KeyError:
            # If key doesn't exist, raise AttributeError
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def recursive_to_dict(d):
    """Recursively convert EasyDict or dict to a standard dict."""
    if isinstance(d, EasyDict):
        d = dict(d)
    if isinstance(d, dict):
        return {k: recursive_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list | tuple):
        return [recursive_to_dict(x) for x in d]
    else:
        return d
