from __future__ import annotations
from typing import Any, Generic, TypeVar, Optional, TypedDict
from dataclasses import dataclass
import warnings

import numpy as np
import copy

from pchandler.base_types import ValidatorsT

T = TypeVar('T')


class FieldOptionsType(TypedDict, Generic[T]):
    type_: type[T]
    optional: bool
    coerce: bool
    freezable: bool
    default: Optional[T]
    validators: ValidatorsT
    _frozen: bool


@dataclass
class FieldOptions(Generic[T]):
    type_: type[T]
    optional: bool = False
    coerce: bool = False
    freezable: bool = False
    default: Optional[T] = None
    validators: ValidatorsT = None
    _frozen: bool = False

    def __post_init__(self):
        self.validators = self.validators or []

class ValidatedField(Generic[T]):
    def __init__(self,
                 type_: type[T],
                 optional: bool = False,
                 coerce: bool = False,
                 freezable: bool = False,
                 default: Optional[T] = None,
                 validators: ValidatorsT = None):
        self._options: FieldOptions = FieldOptions(type_, optional, coerce, freezable, default, validators)

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"
        self.options_name = f"_{name}" + "_options"

    def _get_instance_options(self, instance: object) -> FieldOptions:
        if not hasattr(instance, self.options_name):
            setattr(instance, self.options_name, copy.deepcopy(self._options))
        return getattr(instance, self.options_name)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)

    def __set__(self, instance, value: Optional[T]):
        opts: FieldOptions = self._get_instance_options(instance)

        self.check_frozen(opts)
        value = self.check_optional(value, opts)

        if value is not None:
            value = self.check_coercion(value, opts)
            self.check_type(value, opts.type_)
            self.validate(value, opts)

        setattr(instance, self.private_name, value)

        self.freeze(value, opts)

    def freeze(self, value, opts):
        if opts.freezable:
            opts._frozen = True

    def check_frozen(self, opts: FieldOptions):
        if opts.freezable and opts._frozen:
            raise AttributeError(f"{self.name} is write-once and cannot be modified")

    def check_optional(self, value: T, opts: FieldOptions) -> Optional[T]:
        if value is None:
            if not opts.optional and opts.default is None:
                raise ValueError(f"{self.name} is required and cannot be None")

            if opts.default:
                warnings.warn(f"Setting attribute '{self.name}' to the default of {opts.default} rather than None")
                value = opts.default
            else:
                return None
        return value

    @staticmethod
    def check_coercion(value: Any, opts: FieldOptions) -> T:
        if opts.coerce:
            try:
                value = opts.type_(value)
            except Exception:
                raise TypeError(f"Cannot coerce type {type(value)} to {opts.type_.__name__}")
        return value

    @staticmethod
    def check_type(value: T, type_: type[T]):
        if not isinstance(value, type_):
            raise TypeError(f"Input value: {value} is not of type {type_.__name__}")

    @staticmethod
    def validate(value: T, opts: FieldOptions):
        for validator in opts.validators:
            validator(value)

    def __delete__(self, instance):
        # TODO could be argued if this should be included as the behaviour is not clear. E.g. parameters are erased
        opts = self._get_instance_options(instance)
        if opts.freezable and opts._frozen:
            raise ValueError(f"Attribute '{self.name}' is frozen and cannot be deleted")

        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)



class ValidatedNumpyField(ValidatedField):
    __ndim__: int | None = None
    __shape__: tuple[Optional[int], ...] | None = None
    __dtype__: np.dtype = None

    def __init__(self,
                 optional: bool = False,
                 coerce: bool = False,
                 freezable: bool = False,
                 default: Optional[T] = None,
                 validators: ValidatorsT = None):
        type_ = np.ndarray
        super().__init__(type_, optional, coerce, freezable, default, validators)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self.nd_array_name = self.private_name + "_ndarray"

    @staticmethod
    def check_coercion(value: Any, opts: FieldOptions) -> T:
        if opts.coerce:
            if isinstance(value, (tuple, list, np.ndarray)):
                value = np.asarray(value)
            else:
                raise TypeError(f"Cannot coerce type {type(value)} to {opts.type_.__name__}")
        return value

    def validate(self, value: np.ndarray, opts) -> np.ndarray:
        super().validate(value, opts)
        self._check_ndarray(value)
        return value

    def freeze(self, value: np.ndarray, opts):
        if opts.freezable:
            value.setflags(write=False)
        super().freeze(value, opts)


    def _check_ndarray(self, value: np.ndarray):
        if self.__dtype__ is not None and self.__dtype__ != value.dtype:
            raise TypeError(f"Input array does not have the expected type of {self.__dtype__}")

        if self.__ndim__ is not None and self.__ndim__ != value.ndim:
            raise ValueError(f"Expected array with {self.__ndim__} dimensions. Received [{value.shape=}].")

        if self.__shape__ is not None:
            for i, size_dim in enumerate(self.__shape__):
                if size_dim is not None and size_dim != value.shape[i]:
                    raise ValueError(f"Dimension {i} shape does not match, expected {size_dim} != {value.shape[i]}")


# class ValidatedFieldMeta(type):
#     __field_base_classes__ = ['ValidatedField']
#
#     def __new__(mcs, name, bases, namespace):
#         _annotations = namespace.get('__annotations__', {})
#         fields = {}
#
#         # Create the class first
#         cls = super().__new__(mcs, name, bases, namespace)
#
#         # Process any Field definitions in annotations
#         for key, annotation in _annotations.items():
#             for field_base in mcs.__field_base_classes__:
#                 if isinstance(annotation, str) and field_base in annotation:
#                     try:
#                         field = eval(annotation)
#                         field.__set_name__(cls, key)
#                         setattr(cls, key, field)
#                     except Exception as e:
#                         raise TypeError(f"Failed to create field for {key}: {e}") from e
#         return cls
