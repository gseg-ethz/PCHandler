# from __future__ import annotations
#
# from abc        import  ABC, abstractmethod
# from functools  import  cached_property
# from enum       import  IntEnum
# from typing     import  Literal, Any, overload
#
# import numpy        as  np
# import numpy.typing as  npt
# from trimesh.primitives import Sphere
#
# from pchandler.base_descriptors import ValidatedArrayNx3, PointSet3dAttribute, FieldOptions, ValidationDescriptor, \
#     Point3DAttribute, TransformMatrixDescriptor
# from pchandler.util import bypass_immutable
# from pchandler.base_classes import DataArrayNx3
# from pchandler.geometry.validation import check_spherical_coordinates
# from pchandler.types import NumOrArray
#
#
# class CoordinatesF32Descriptor(PointSet3dAttribute):
#     __dtype__ = np.float32
#
# class CoordinatesF64Descriptor(PointSet3dAttribute):
#     __dtype__ = np.float64
#
#
#
#
# class Abstract3DCoordinates(ABC, ValidatedArrayNx3):
#     _arr: CoordinatesF32Descriptor = CoordinatesF32Descriptor(coerce=True)
#     _shift: Point3DAttribute = Point3DAttribute(optional=False, default=np.zeros(3).astype(np.float32))
#     _project_transform: TransformMatrixDescriptor = TransformMatrixDescriptor(optional=False)
#     _global_shifted: ValidationDescriptor = ValidationDescriptor(bool, default=False, optional=False)
#
#     def __init__(self,
#                  coordinates: np.ndarray,
#                  shift: np.ndarray = np.zeros(3, dtype=np.float32),
#                  transform_matrix: np.ndarray | TransformMatrixAttribute = np.eye(4, dtype=np.float32),
#                  global_shifted: bool = False) -> None:
#         self._arr = coordinates
#         self._shift = shift
#         self._project_transform = TransformMatrixAttribute(transform_matrix)
#         self._global_shifted = global_shifted
#
# # TODO the idea here is that whenever initialised, the scan center coordinates are always provided
# #  A transformation can be provided with it
# class CartesianCoordinates(Abstract3DCoordinates):
#
#     # TODO implement a function to gracefully handle transformation matrices
#
#     # TODO discuss - my idea is that much like in cloud compare, the shift is applied on load and then never updated.
#     #  well... I assume it works that way
#     def as_global(self) -> CartesianCoordinates:
#         return self.as_local() @ self._project_transform
#
#     def as_local(self) -> CartesianCoordinates:
#         if self._global_shifted:
#             return self + self._shift
#         else:
#             return self
#
#     def as_optimal(self) -> CartesianCoordinates:
#         return self._arr
#
#     def coerce_array(self, xyz: np.ndarray) -> np.ndarray:
#         arr, self._shift, self._global_shifted = self.compute_global_shift(xyz)
#         return arr
#
#     def compute_global_shift(self, xyz: np.ndarray, decimal_magnitude: int = 4) -> tuple[np.ndarray, np.ndarray, bool]:
#         if self._is_shift_needed(xyz):
#             offset: np.ndarray = np.median( np.round( xyz, decimals = -( decimal_magnitude - 1 ) ), axis = 0 )
#             xyz = (xyz - self._shift).astype(np.float32)
#             return xyz-offset, offset, True
#         else:
#             return xyz.astype(np.float32), np.zeros(3).astype(np.float32), False
#
#     def __setitem__(self, key: str, value: np.ndarray):
#         super().__setitem__(key, value)
#         if key == 'arr':
#             del self.spher
#
#     @property
#     def x(self): return self._arr[:, 0]
#
#     @property
#     def y(self): return self._arr[:, 1]
#
#     @property
#     def z(self): return self._arr[:, 2]
#
#     @property
#     def xyz(self): return self._arr
#
#     @property
#     def yxz(self): return np.vstack((self.y, self.x, self.z)).T
#
#     @staticmethod
#     def _is_shift_needed(xyz: np.ndarray, decimal_magnitude: int = 4) -> np.bool_:
#         return np.any(np.abs(xyz) >= 10 ** decimal_magnitude)
#
#     def to_spherical(self) -> SphericalCoordinates:
#         return cartesian2spherical(self.as_local())
#
#     @classmethod
#     def from_spherical(cls, spherical: SphericalCoordinates) -> CartesianCoordinates:
#         return spherical2cartesian(spherical)
#
#     @cached_property
#     def spher(self) -> SphericalCoordinates:
#         return self.to_spherical()
#
#     @cached_property
#     def r(self):
#         return self.spher.r
#
#     @property
#     def hz(self):
#         return self.spher.hz
#
#     @property
#     def v(self):
#         return self.spher.v
#
#
# class SphericalCoordinates(AbstractCoordinates):
#     @abstractmethod
#     @property
#     def r(self) -> np.ndarray:
#         pass
#
#     @abstractmethod
#     @property
#     def hz(self) -> np.ndarray:
#         pass
#
#     @abstractmethod
#     @property
#     def v(self):
#         pass
#
#     @abstractmethod
#     def to_cartesian
#
#
#
#
#
# def spherical2cartesian(coordinates: SphericalCoordinates) -> np.ndarray|tuple[np.ndarray, np.ndarray, np.ndarray]:
#
#     xyz = np.zeros_like(coordinates)
#     xyz[:, 0] = coordinates.r * np.sin(coordinates.v) * np.cos(coordinates.hz)
#     xyz[:, 1] = coordinates.r * np.sin(coordinates.v) * np.sin(coordinates.hz)
#     xyz[:, 2] = coordinates.r * np.cos(coordinates.v)
#
#     coordinates.arr = xyz
#
#
#
# def spher2cart_vec(rho: npt.ArrayLike, theta: npt.ArrayLike, phi: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
#
#     return x, y, z
#
# class CoordSysEnum(IntEnum):
#     CART = 0
#     SPHER = 1
#
#
# def spherical2cartesian(spherical: npt.NDArray) -> npt.NDArray:
#     xyz: npt.NDArray = np.zeros_like(spherical)
#     xyz[:, 0], xyz[:, 1], xyz[:, 2] = spher2cart_vec(spherical[:, 0], spherical[:, 1], spherical[:, 2])
#     return xyz
#
#
# def cartesian2spherical(xyz: CartesianCoordinates) -> SphericalCoordinates:
#     spherical[:, 0], spherical[:, 1], spherical[:, 2] = cart2spher_vec(xyz[:, 0], xyz[:, 1], xyz[:, 2])
#     return spherical
#
# def cart2spher_vec(xyz: CartesianCoordinates) -> SphericalCoordinates:
#     rhv: np.ndarray = np.zeros_like(xyz)
#     xy_2: np.ndarray = xyz.x ** 2 + xyz.y ** 2
#     rhv[:, 0] = np.sqrt(xy_2 + xyz.z ** 2)              # [  0, inf] slope distance
#     rhv[:, 2] = np.arctan2(np.sqrt(xy_2), xyz.z)        # [  0, +pi] zenith angle
#     rhv[:, 1] = np.arctan2(xyz.y, xyz.x)                 # [-pi, +pi] horizonal angle
#     return rho, theta, phi
#
#
#
# # TODO Implement a Storage and View Structure design
# #  - separation of concerns
# #  - shallow call paths
# #  - coordinate system control
# #  - caching of the complimentary coordinates
# #  - cleanm descriptive code
# #  - no need to recompute coordinates if not required
# #  - mutability in place will depend on if the overarching array is provided as mutable
# #  - invalidate cache on mutation
# class CoordinateSet3D(DataArrayNx3):
#     coord_system: CoordSysEnum = CoordSysEnum.CART
#
#     def __init__(self,
#                  array,
#                  coord_system: Literal['cartesian']|Literal['spherical']|CoordSysEnum = CoordSysEnum.CART,
#                  **kwargs):
#
#         super().__init__(array, **kwargs)
#         if coord_system not in CoordSysEnum:
#             raise ValueError("coord_system must be from 'CoordSysEnum'")
#
#         self.coord_system = CoordSysEnum[coord_system] if isinstance(coord_system, str) else coord_system
#
#     @property
#     def num_points(self) -> int:
#         return len(self)
#
#     @property
#     def _prop_names(self) -> frozenset[str]:
#         return frozenset([])
#
#     def invalidate_cache(self):
#         for name in self._prop_names:
#             if name in self.__dict__:
#                 del self.__dict__[name]
#
#     def _convert_to_system(self, target_system: CoordSysEnum):
#         if self.coord_system != target_system:
#
#             self.invalidate_cache()
#             self.coord_system = target_system
#
#             if self.coord_system == CoordSysEnum.CART:
#                 self.arr = spherical2cartesian(self.arr)
#             else:
#                 self.arr = cartesian2spherical(self.arr)
#
#     def validate(self, array: np.ndarray) -> np.ndarray:
#         if not np.issubdtype(array.dtype, np.floating):
#             raise TypeError(f"Expected floating point array. Received {array.dtype}.")
#
#         if self.coord_system == CoordSysEnum.SPHER:
#             check_spherical_coordinates(array)
#         return array
#
#
# class CartesianCoordinates(CoordinateSet3D):
#     @property
#     def _prop_names(self) -> frozenset[str]:
#         return super()._prop_names | frozenset({'_xyz'})
#
#     @cached_property
#     def _xyz(self) -> np.ndarray:
#         return spherical2cartesian(self.arr)
#
#     @property
#     def xyz(self) -> np.ndarray:
#         return self._get_cartesian_data()
#
#     @xyz.setter
#     def xyz(self, xyz: np.ndarray) -> None:
#         if self.coord_system != CoordSysEnum.CART:
#             self._convert_to_system(CoordSysEnum.CART)
#         self.arr = xyz
#
#     @property
#     def x(self) -> NumOrArray:
#         return self.xyz[:, 0].view()
#
#     @x.setter
#     def x(self, x: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.CART:
#             raise ValueError("Cannot set 'x' whilst coord system is SPHERICAL")
#         self.xyz[:, 0] = x
#
#     @property
#     def y(self) -> NumOrArray:
#         return self.xyz[:, 1].view()
#
#     @y.setter
#     def y(self, y: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.CART:
#             raise ValueError("Cannot set 'y' whilst coord system is SPHERICAL")
#         self.xyz[:, 1] = y
#
#     @property
#     def z(self) -> NumOrArray:
#         return self.xyz[:, 2].view()
#
#     @z.setter
#     def z(self, z: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.CART:
#             raise ValueError("Cannot set 'z' whilst coord system is SPHERICAL")
#         self.xyz[:, 2] = z
#
#     def _get_cartesian_data(self):
#         if self.coord_system == CoordSysEnum.SPHER:
#             return self._xyz
#         return self.arr.view()
#
#     @bypass_immutable
#     def to_spherical(self):
#         self._convert_to_system(CoordSysEnum.SPHER)
#
#
# class SphereCoordinates(CoordinateSet3D):
#     @property
#     def _prop_names(self) -> frozenset[str]:
#         return super()._prop_names | frozenset(['_spher'])
#
#     @property
#     def spher(self) -> np.ndarray:
#         return self._get_spherical_data()
#
#     @spher.setter
#     def spher(self, spher: np.ndarray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             self._convert_to_system(CoordSysEnum.SPHER)
#         self.arr = spher
#
#
#     @cached_property
#     def _spher(self) -> np.ndarray:
#         return cartesian2spherical(self.arr)
#
#     @property
#     def r(self):
#         return self.spher[:, 0].view()
#
#     @r.setter
#     def r(self, r: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'r' whilst coord system is CARTESIAN")
#         self.spher[:, 0] = r
#
#     @property
#     def v(self):
#         return self.spher[:, 1].view()
#
#     @v.setter
#     def v(self, v: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'v' whilst coord system is SPHERICAL")
#         self.spher[:, 1] = v
#
#     @property
#     def hz(self):
#         return self.spher[:, 2].view()
#
#     @hz.setter
#     def hz(self, hz: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'hz' whilst coord system is SPHERICAL")
#         self.spher[:, 2] = hz
#
#     @property
#     def rho(self):
#         return self.r
#
#     @rho.setter
#     def rho(self, rho: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'rho' whilst coord system is SPHERICAL")
#         self.spher[:, 0] = rho
#
#     @property
#     def theta(self):
#         return self.v
#
#     @theta.setter
#     def theta(self, theta: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'theta' whilst coord system is SPHERICAL")
#         self.spher[:, 1] = theta
#
#     @property
#     def phi(self):
#         return self.hz
#
#     @phi.setter
#     def phi(self, phi: NumOrArray) -> None:
#         if self.coord_system != CoordSysEnum.SPHER:
#             raise ValueError("Cannot set 'phi' whilst coord system is SPHERICAL")
#         self.spher[:, 2] = phi
#
#     def _get_spherical_data(self):
#         if self.coord_system == CoordSysEnum.CART:
#             return self._spher
#         return self.arr.view()
#
#     @bypass_immutable
#     def to_cartesian(self):
#         self._convert_to_system(CoordSysEnum.CART)
#
#
# class GeneralCoordinates(CartesianCoordinates, SphereCoordinates):
#     @property
#     def _prop_names(self):
#         return super()._prop_names