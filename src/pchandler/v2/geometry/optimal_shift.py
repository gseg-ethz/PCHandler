from __future__ import annotations

import weakref
from enum import member
from typing import NamedTuple, Self, TypeVar, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import PointCloudData


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class OptimizedShiftManager(metaclass=SingletonMeta):
    _optimized_shifts: weakref.WeakSet[OptimizedShift]

    def __init__(self) -> None:
        self._optimized_shifts = weakref.WeakSet()

    def _register(self, shift: OptimizedShift) -> None:
        self._optimized_shifts.add(shift)

    def get_new(self) -> OptimizedShift:
        return OptimizedShift()


class OptimizedShift:
    optimal_shift: NDArray[np.float64]
    member_pcds: weakref.WeakSet["PointCloudData"]

    def __init__(self) -> None:
        self.optimal_shift = np.zeros(3)
        self.member_pcds = weakref.WeakSet()
        OptimizedShiftManager()._register(self)


    def register(self, pcd: "PointCloudData") -> None:
        self.member_pcds.add(pcd)



# T = TypeVar('T')
#
#
# class OptimalShiftManager[T]:
#     """
#     Singleton object following enabling an observer pattern for optimal shift changes
#     """
#     optimal_shift: np.ndarray = np.zeros(3)
#     current_bbox: np.ndarray|None = None
#     bounding_boxes: np.ndarray|None = None
#     point_clouds = weakref.WeakSet()
#     __instance: Self|None = None
#
#     def __new__(cls, *args, **kwargs) -> OptimalShiftManager[T]:
#         if cls.__instance is None:
#             cls.__instance = super().__new__(cls)
#         return cls.__instance
#
#     def __init__(self, *args, **kwargs) -> None: ...
#
#     @classmethod
#     def reset(cls):
#         cls.current_bbox = None
#         cls.bounding_boxes = None
#         cls.optimal_shift = np.zeros(3)
#         cls.point_clouds = weakref.WeakSet()
#
#     @classmethod
#     def register(cls, pcd: T) -> None:
#         """Register new point clouds and keep record of the original bounding box locations"""
#         if cls.bounding_boxes is None:
#             cls.bounding_boxes = (
#                 np.vstack((pcd.min(axis=0), pcd.max(axis=0))))
#         else:
#             cls.bounding_boxes = (
#                 np.vstack((cls.bounding_boxes, pcd.min(axis=0), pcd.max(axis=0))))
#
#         # There is only ever one project bounding box.
#         # When it becomes a problem and wanting to manage two coord systems, then it's time to adapt the code
#         pcd.arr = (pcd.arr - (cls.optimal_shift.astype(np.float32)))
#         cls.point_clouds.add(pcd)
#         print(f'PCD added - {id(pcd)=}')
#
#     @classmethod
#     def update_optimal_shift(cls):
#         # DISCUSS it appears to work...
#         shift = cls.compute_shift(cls.bounding_boxes)
#         delta_shift = shift - cls.optimal_shift
#         cls.optimal_shift = shift
#
#         for pcd in cls.point_clouds:
#             pcd.arr -= delta_shift
#             pcd.socs_origin -= delta_shift
#             pcd.is_optimised = True
#
#         # Verify no extra shift needed and end up in an infinite loop due to too large bbox
#         cls.current_bbox = cls._get_current_optimal_bbox()
#         if cls._is_shift_needed(cls.current_bbox):
#             raise ValueError('The point clouds no longer fit in the optimal shift range after the last one added')
#
#     @classmethod
#     def _get_current_optimal_bbox(cls):
#         return np.vstack(
#             (cls.bounding_boxes.min(axis=0), cls.bounding_boxes.max(axis=0))
#         ) - cls.optimal_shift
#
#     @classmethod
#     def _get_all_limits(cls) -> np.ndarray:
#         minima = np.zeros((len(cls.point_clouds), 3), dtype=np.float64)
#         maxima = minima.copy()
#
#         for i, pcd in enumerate(cls.point_clouds):
#             minima[i, :] = pcd.min(axis=0)
#             maxima[i, :] = pcd.max(axis=0)
#
#         return np.vstack((minima, maxima))
#
#     @staticmethod
#     def _is_shift_needed(limits: np.ndarray, decimal_magnitude: int = 4) -> np.bool_:
#         return np.any(np.abs(limits) >= 10 ** decimal_magnitude)
#
#     @staticmethod
#     def compute_shift(limit_points: np.ndarray, decimal_magnitude: int = 4) -> np.ndarray:
#         return np.median(np.round(limit_points, decimals=-(decimal_magnitude - 1)), axis=0)
#
#     @classmethod
#     def register_point_cloud(cls, target_cls):
#         original_init = target_cls.__init__
#
#         def new_init(self, *args, **kwargs):
#             original_init(self, *args, **kwargs)
#             if kwargs.get('optimal', False):
#                 cls().register(self)
#                 cls.update_optimal_shift()
#
#         target_cls.__init__ = new_init
#         return target_cls
#
# OSM_Manager = OptimalShiftManager()
