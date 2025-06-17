from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, ClassVar, Self, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import PointCloudData

from ..base_types import Vector_3_T, Array_Nx3_T

class SingletonMeta(type):
    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args, **kwargs) -> object:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class OptimizedShiftManager(metaclass=SingletonMeta):
    _optimized_shifts: weakref.WeakSet[OptimizedShift]
    _maximum_decimal_places: int

    def __init__(self, maximum_decimal_places: int = 4) -> None:
        self._optimized_shifts = weakref.WeakSet()
        self._maximum_decimal_places = maximum_decimal_places

    def register(self, shift: OptimizedShift) -> None:
        self._optimized_shifts.add(shift)

    @staticmethod
    def new_shift() -> OptimizedShift:
        return OptimizedShift()

    @property
    def all_shifts(self) -> list[OptimizedShift]:
        return list(self._optimized_shifts)

    @property
    def maximum_decimal_places(self) -> int:
        return self._maximum_decimal_places

    @maximum_decimal_places.setter
    def maximum_decimal_places(self, maximum_decimal_places: int) -> None:
        pass # Todo: Should probably check all registered GlobalShifts and their pcds if they conform to new limit

    def is_shift_needed(self, values: NDArray[np.floating] | "PointCloudData") -> bool:
        return np.any(np.abs(values) >= 10 ** self._maximum_decimal_places)

class OptimizedShift:

    class MinMaxPoints(NamedTuple):
        minimum: Vector_3_T
        maximum: Vector_3_T

        @classmethod
        def from_points(cls, points: Array_Nx3_T) -> Self:
            min_point = np.min(points, axis=0)
            max_point = np.max(points, axis=0)
            return cls(min_point, max_point)

        @property
        def central_point(self) -> Vector_3_T:
            return Vector_3_T(np.mean(np.vstack((self.minimum, self.maximum)), axis=0))



    _optimal_shift: Vector_3_T
    _member_pcds: weakref.WeakSet["PointCloudData"]
    _member_pcds_bbox: weakref.WeakKeyDictionary["PointCloudData", MinMaxPoints]

    def __init__(self) -> None:
        self._optimal_shift = Vector_3_T(np.zeros(3))
        self._member_pcds = weakref.WeakSet()
        self._member_pcds_bbox = weakref.WeakKeyDictionary()
        OptimizedShiftManager().register(self)

    def register(self, pcd: "PointCloudData", values: Array_Nx3_T) -> Vector_3_T:
        if self.does_current_shift_fit(values):
            self._member_pcds.add(pcd)
            self._member_pcds_bbox[pcd] = OptimizedShift.MinMaxPoints.from_points(values)
            return self._optimal_shift
        if len(self._member_pcds) == 0:
            self._optimal_shift = self.calculate_shift(pcd)
            self._member_pcds.add(pcd)
            return


        self._member_pcds.add(pcd)



    @property
    def optimal_shift(self) -> NDArray[np.float64]:
        return self._optimal_shift


    def does_current_shift_fit(self, values: Array_Nx3_T | NDArray[np.floating]) -> bool:
        return not bool(OptimizedShiftManager().is_shift_needed(np.subtract(values, self._optimal_shift)))

    def calculate_new_shift(self, values: Array_Nx3_T| NDArray[np.floating]) -> Vector_3_T:
        minmax = OptimizedShift.MinMaxPoints.from_points(values)



    # def
    # def calculate_shift(self, pcd: "PointCloudData") -> NDArray[np.float64]:

    @staticmethod
    def calculate_shift(values: NDArray[np.floating] | "PointCloudData") -> Array_Nx3_T:
        return Array_Nx3_T(np.median(np.round(values, decimals=-(OptimizedShiftManager().maximum_decimal_places - 1)), axis=0).astype(np.float64))



#
#
# # T = TypeVar('T')
# #
# #
# # class OptimalShiftManager[T]:
# #     """
# #     Singleton object following enabling an observer pattern for optimal shift changes
# #     """
# #     optimal_shift: np.ndarray = np.zeros(3)
# #     current_bbox: np.ndarray|None = None
# #     bounding_boxes: np.ndarray|None = None
# #     point_clouds = weakref.WeakSet()
# #     __instance: Self|None = None
# #
# #     def __new__(cls, *args, **kwargs) -> OptimalShiftManager[T]:
# #         if cls.__instance is None:
# #             cls.__instance = super().__new__(cls)
# #         return cls.__instance
# #
# #     def __init__(self, *args, **kwargs) -> None: ...
# #
# #     @classmethod
# #     def reset(cls):
# #         cls.current_bbox = None
# #         cls.bounding_boxes = None
# #         cls.optimal_shift = np.zeros(3)
# #         cls.point_clouds = weakref.WeakSet()
# #
# #     @classmethod
# #     def register(cls, pcd: T) -> None:
# #         """Register new point clouds and keep record of the original bounding box locations"""
# #         if cls.bounding_boxes is None:
# #             cls.bounding_boxes = (
# #                 np.vstack((pcd.min(axis=0), pcd.max(axis=0))))
# #         else:
# #             cls.bounding_boxes = (
# #                 np.vstack((cls.bounding_boxes, pcd.min(axis=0), pcd.max(axis=0))))
# #
# #         # There is only ever one project bounding box.
# #         # When it becomes a problem and wanting to manage two coord systems, then it's time to adapt the code
# #         pcd.arr = (pcd.arr - (cls.optimal_shift.astype(np.float32)))
# #         cls.point_clouds.add(pcd)
# #         print(f'PCD added - {id(pcd)=}')
# #
# #     @classmethod
# #     def update_optimal_shift(cls):
# #         # DISCUSS it appears to work...
# #         shift = cls.compute_shift(cls.bounding_boxes)
# #         delta_shift = shift - cls.optimal_shift
# #         cls.optimal_shift = shift
# #
# #         for pcd in cls.point_clouds:
# #             pcd.arr -= delta_shift
# #             pcd.socs_origin -= delta_shift
# #             pcd.is_optimised = True
# #
# #         # Verify no extra shift needed and end up in an infinite loop due to too large bbox
# #         cls.current_bbox = cls._get_current_optimal_bbox()
# #         if cls._is_shift_needed(cls.current_bbox):
# #             raise ValueError('The point clouds no longer fit in the optimal shift range after the last one added')
# #
# #     @classmethod
# #     def _get_current_optimal_bbox(cls):
# #         return np.vstack(
# #             (cls.bounding_boxes.min(axis=0), cls.bounding_boxes.max(axis=0))
# #         ) - cls.optimal_shift
# #
# #     @classmethod
# #     def _get_all_limits(cls) -> np.ndarray:
# #         minima = np.zeros((len(cls.point_clouds), 3), dtype=np.float64)
# #         maxima = minima.copy()
# #
# #         for i, pcd in enumerate(cls.point_clouds):
# #             minima[i, :] = pcd.min(axis=0)
# #             maxima[i, :] = pcd.max(axis=0)
# #
# #         return np.vstack((minima, maxima))
# #
# #     @staticmethod
# #     def _is_shift_needed(limits: np.ndarray, decimal_magnitude: int = 4) -> np.bool_:
# #         return np.any(np.abs(limits) >= 10 ** decimal_magnitude)
# #
# #     @staticmethod
# #     def compute_shift(limit_points: np.ndarray, decimal_magnitude: int = 4) -> np.ndarray:
# #         return np.median(np.round(limit_points, decimals=-(decimal_magnitude - 1)), axis=0)
# #
# #     @classmethod
# #     def register_point_cloud(cls, target_cls):
# #         original_init = target_cls.__init__
# #
# #         def new_init(self, *args, **kwargs):
# #             original_init(self, *args, **kwargs)
# #             if kwargs.get('optimal', False):
# #                 cls().register(self)
# #                 cls.update_optimal_shift()
# #
# #         target_cls.__init__ = new_init
# #         return target_cls
# #
# # OSM_Manager = OptimalShiftManager()
