import sys

from dataclasses import dataclass, field, InitVar, KW_ONLY
from collections.abc import MutableMapping
import logging
from typing import Iterator, Optional, Iterable
if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray, DTypeLike

logger = logging.getLogger(__name__.split(".")[0])

@dataclass(frozen=True)
class ScalarField:
    """
    Represents a scalar field associated with a point cloud.

    Attributes
    ----------
    name : str
        The name of the scalar field.
    data : NDArray[np.generic]
        A 1D numpy array containing scalar data (one value per point).
    original_dtype : Optional[DTypeLike]
        Optional unit for the scalar field (e.g., "m", "intensity").
    operations_performed : Optional[str]
        Optional description providing additional context for the scalar field.
    """
    name: str
    data: NDArray[np.float32]
    _: KW_ONLY
    original_dtype: Optional[DTypeLike] = None
    operations_performed: list[tuple[str, tuple[...]]] = field(default_factory=list)
    override_forced_dtype_conversion: InitVar[bool] = False

    def __post_init__(self, override_forced_dtype_conversion: bool) -> None:
        if not isinstance(self.name, str):
            raise TypeError("ScalarField name must be a string")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("ScalarField values must be a numpy array")
        if self.data.ndim != 1:
            raise ValueError("ScalarField values must be a 1D numpy array")
        if self.data.dtype != np.float32 and not override_forced_dtype_conversion:
            logger.debug(f"Scalar field `{self.name}` converted from {self.data.dtype} to float32")
            object.__setattr__(self, "original_dtype", self.data.dtype)
            self.operations_performed.append(("dtype_conversion", (self.data.dtype, np.float32)))
            object.__setattr__(self, "data", self.data.astype(np.float32))
        elif self.data.dtype != np.float32 and override_forced_dtype_conversion:
            logger.warning(f"Scalar field `{self.name}` not converted to float32. This may not be fully supported."
                           f"Use at your own risk!")

    def __getitem__(self, key: [slice, NDArray[np.bool_], NDArray[np.int_], list]) -> Self:
        """
        Returns a new ScalarField with values indexed by key.
        Depending on the type of key, this may return a view or a copy.
        """
        new_values = self.data.copy()[key]
        return ScalarField(
            name=self.name,
            data=new_values,
            original_dtype=self.original_dtype,
            operations_performed=self.operations_performed.copy()
        )

    @property
    def __array_interface__(self) -> dict:
        """
        Function for direct numpy interoperability of ScalarField.
        """
        return self.data.__array_interface__

    # def __array__(self, dtype: Optional[DTypeLike]=None, copy: Optional[bool]=None) -> NDArray[np.generic]:
    #     """
    #     Function for direct numpy interoperability of ScalarField.
    #     Parameters
    #     ----------
    #     dtype: Optional[DTypeLike]
    #     copy: Optional[bool]
    #
    #     Returns
    #     -------
    #
    #     """
    #     data = self.data
    #     if dtype is not None and self.data.dtype != dtype:
    #         data = data.astype(dtype)
    #     if copy is not None:
    #         data = data.copy()
    #     return data

    def __len__(self) -> int:
        return self.data.shape[0]

    def normalize(self, lower: float = None, upper: float = None) -> None:
        lower = self.data.min() if lower is None else lower
        upper = self.data.max() if upper is None else upper

        assert lower < upper
        # self.data = self.data / (self.data.max() - self.data.min())
        np.divide(self.data - lower, upper-lower, out=self.data)
        self.operations_performed.append(("normalize", (lower, upper)))
        logger.debug(f"Normalized scalar field `{self.name}` from (original) span [{lower}, {upper}] to [0, 1].")

    def normalize_based_on_original_dtype(self) -> None:
        if self.original_dtype is None:
            logger.debug(f"Scalar field `{self.name}` wasn't converted. No operation performed.")
            return
        if self.original_dtype.kind not in ["u", "i"]:
            logger.debug(f"Scalar field `{self.name}` was originally a float. No operation performed.")
            return

        lower = np.iinfo(self.original_dtype).min
        upper = np.iinfo(self.original_dtype).max
        self.normalize(lower=lower, upper=upper)

    def create_rollback(self) -> NDArray[np.generic]:
        data = self.data.copy()
        for operation, operation_parameters in self.operations_performed[::-1]:
            match operation:
                case "normalize":
                    lower, upper = operation_parameters
                    np.multiply(self.data, upper-lower, out=data)
                    np.add(data, lower, out=data)
                case "dtype_conversion":
                    data = data.astype(operation_parameters[0])
                case _:
                    return ValueError(f"Operation {operation} not supported.")
        assert self.original_dtype is None or data.dtype == self.original_dtype

        logger.debug(f"Converted scalar field `{self.name}` to original bounds and dtype.")
        return data



class ScalarFieldManager(MutableMapping):
    """
    Manages a collection of ScalarField objects, ensuring that all fields have the same
    number of data points. Also provides a mechanism to select subsets of the fields.
    """
    def __init__(self, expected_length: Optional[int] = None):
        self._fields: dict[str, ScalarField] = {}
        self._expected_length: Optional[int] = expected_length

    def __getitem__(self, key: [str, slice, NDArray[np.bool_], NDArray[np.int_], list]) -> [ScalarField, Self]:
        # If key is a string, return the corresponding field.
        if isinstance(key, str):
            return self._fields[key.lower()]
        # Otherwise, assume we are slicing across all fields.
        new_manager = ScalarFieldManager()
        for sf_field in self._fields.values():
            new_manager.add_field(sf_field[key])
        return new_manager

    def __setitem__(self, key: str, value: ScalarField | NDArray):
        if not isinstance(key, str):
            raise TypeError("ScalarField key must be a string")
        key = key.lower()
        if not isinstance(value, ScalarField) and not isinstance(value, np.ndarray):
            raise TypeError("Value must be an instance of ScalarField or NDArray")
        if isinstance(value, ScalarField) and value.name.lower() != key:
            raise ValueError("ScalarField name and key do not match")

        if isinstance(value, np.ndarray):
            value = ScalarField(name=key, data=value)

        field_length = value.data.shape[0]
        if self._expected_length is None:
            self._expected_length = field_length
        elif field_length != self._expected_length:
            raise ValueError(
                f"All scalar fields must have the same number of data points. "
                f"Expected {self._expected_length}, got {field_length}"
            )
        self._fields[key] = value

    @property
    def shape(self) -> tuple[int, ...]:
        # if len(self) == 0:
        #     return (0,self._expected_length)
        return (len(self), self._expected_length,)


    def __delitem__(self, key: str):
        del self._fields[key]
        if not self._fields:
            self._expected_length = None

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def add_field(self, sf_field: ScalarField) -> None:
        self[sf_field.name.lower()] = sf_field

    def create_field(self, name: str, data: NDArray[np.generic]) -> None:
        sf = ScalarField(name=name, data=data)
        self.add_field(sf)

    def remove_field(self, field_name: str) -> None:
        del self[field_name.lower()]

    def keys(self) -> list[str]:
        """
        Returns a list of all available scalar field keys.
        """
        return list(self._fields.keys())

    def values(self) -> Iterator[ScalarField]:
        """
        Returns an iterator over all scalar fields.
        """
        return iter(self._fields.values())

    def items(self) -> Iterator[tuple[str, ScalarField]]:
        """
        Returns an iterator over key-value pairs of scalar fields.
        """
        return iter(self._fields.items())

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._fields

    @classmethod
    def merge(cls, sfms: Iterable[Self]) -> Self:
        common_keys = set.intersection(*(set(sfm.keys()) for sfm in sfms))
        if len(common_keys) == 0:
            return ScalarFieldManager()

        expected_length = sum([sfm.shape[1] for sfm in sfms])
        new_sfm = ScalarFieldManager(expected_length=expected_length)
        for common_key in common_keys:
            sfs: list[ScalarField] = [sfm[common_key] for sfm in sfms]
            if len(set([sf.name for sf in sfs])) != 1:
                logger.warning(f"While merging scalar field {common_key} different names were encountered.")
            name = sfs[0].name

            if len(set(map(tuple, [sf.operations_performed for sf in sfs]))) != 1:
                logger.warning(f"While merging scalar field {common_key} different list of previously performed "
                               f"operations were encountered. Merged scalar field will have an empty record!")
                operations_performed = []
            else:
                operations_performed = sfs[0].operations_performed
            if len(set(sf.original_dtype for sf in sfs)) != 1:
                logger.warning(f"While merging scalar field {common_key} different original dtypes were encountered. "
                               f"Merged scalar field will have an empty record!")
                original_dtype = None
            else:
                original_dtype = sfs[0].original_dtype

            data = np.concatenate([sf.data for sf in sfs])
            sf = ScalarField(name=name, data=data, original_dtype=original_dtype, operations_performed=operations_performed)
            new_sfm.add_field(sf)

        return new_sfm







