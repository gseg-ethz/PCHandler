from __future__     import annotations


import  warnings
import  copy
from    abc         import ABC, abstractmethod
from    typing      import Any, Generic, TypeVar, Optional, TypedDict, Callable
from    dataclasses import dataclass

import  numpy as np

T = TypeVar('T')
ValidatorsT = list[Callable[[T], None]]


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

# TODO Discuss w/ Nicholas thought's on default parameters here - e.g. should it be most or least flexible to start?
class ValidatedAttribute(Generic[T]):
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


class ValidatedArrayAttribute(ValidatedAttribute):
    # TODO discuss if it's worth extending to create class and instance level definitions
    __ndim__: int | None = None
    __shape__: tuple[Optional[int], ...] = (None,)
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
        # TODO something needs to be done with this or it's removed
        self.nd_array_name = self.private_name + "_ndarray"

    # TODO need to decide on an approach about the type coercion. For example min_scalar_type()
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


class VectorAttribute(ValidatedArrayAttribute):
    # TODO should we put in a function that squeeze's the array incase a vector is passed of another shape (1, 1, N)?
    __ndim__: int = 1


class Array2dAttribute(ValidatedArrayAttribute):
    __ndim__: int = 2


class Array3dAttribute(ValidatedArrayAttribute):
    __ndim__: int = 3


class PointSet2D(Array2dAttribute):
    # TODO Discuss w/ Nicholas about whether to extend this to define and validate this on a per-instance_basis
    __shape__: tuple[Optional[int], ...] = (None, 2)


class PointSet3D(Array3dAttribute):
    __shape__: tuple[Optional[int], ...] = (None, 3)



class ValidatedArray(np.lib.mixins.NDArrayOperatorsMixin):
    arr: ValidatedArrayAttribute = ValidatedArrayAttribute()

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __init__(self, array: np.ndarray | ValidatedArray):

        if isinstance(array, ValidatedArray):
            self.__dict__ = copy.deepcopy(self.__dict__)
        else:
            array = self.coerce(array)
            self.arr: np.ndarray = array

    def __getitem__(self, index: Any) -> np.ndarray:
        return self.arr[index]

    def __setitem__(self, index, value: np.ndarray | float | int | bool) \
            -> ValidatedArray | np.ndarray | float | int | bool:
        self.arr[index] = value[index] if isinstance(value, ValidatedArray) else value

    def __array__(self) -> np.ndarray:
        return self.arr

    @property
    def __array_interface__(self) -> dict:
        return self.arr.__array_interface__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) \
            -> np.ndarray|tuple[np.ndarray,...]|tuple[ValidatedArray,...]:

        arrays = [x.arr if isinstance(x, ValidatedArray) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, tuple):
            return tuple(x if np.issubdtype(x.dtype, np.bool_) else type(self)(x) for x in result)

        elif isinstance(result, np.ndarray):
            return result if np.issubdtype(result.dtype, np.bool_) else type(self)(result)

        else:
            return result

    def coerce(self, array: np.ndarray) -> np.ndarray:
        return array

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> np.dtype:
        return self.arr.dtype

    @property
    def size(self) -> int:
        return self.arr.size

    @property
    def base(self):
        return self.arr.base

    def copy(self, deep: bool = False) -> ValidatedArray:
        return copy.deepcopy(self) if deep else copy.copy(self)

    @property
    def view(self):
        return self.arr.view(type(self))

class ReadOnlyArrayAttribute(ValidatedArray):
    arr: ValidatedArrayAttribute = ValidatedArrayAttribute(freezable=True)

class ValidatedVector(ValidatedArray):
    arr: VectorAttribute = VectorAttribute()

    def coerce(self, value: np.ndarray):
        value = value.squeeze()
        return super().coerce(value)

class ValidatedArray2D(ValidatedArray):
    arr: Array2dAttribute = Array2dAttribute()

class ValidatedArrayNx3(ValidatedArray):
    arr: PointSet3D = PointSet3D()


class ValidatedArrayNx2(ValidatedArray):
    arr: PointSet2D = PointSet2D()


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
