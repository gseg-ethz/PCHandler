from __future__ import annotations
from typing import TypeVar
import numpy as np

T = TypeVar('T')

class OptimalShiftManager[T]:
    """
    Singleton object that will return the single instance on each object creation
    """
    __optimal_shift = np.zeros(3, dtype=np.float64)
    __instance = None
    __pcds = []

    def __new__(cls, pcd: T, *args, **kwargs) -> OptimalShiftManager[T]:
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        cls.register(pcd)
        return cls.__instance

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def register(cls, pcd: T) -> None:
        cls.__instance.__pcds.append(pcd)
        print(f'Object added - {id(pcd)=}')

    @staticmethod
    def is_shift_needed(pcd: T) -> bool:
        raise NotImplementedError()

    @staticmethod
    def compute_shift(pcd: T) -> T:
        raise NotImplementedError()

    @staticmethod
    def compute_project_shift(pcds: list[T]):
        raise NotImplementedError()

    @classmethod
    def check_if_update_needed(cls, pcd) -> bool:
        """Perform check with existing shift and min/max bounds to see if it falls within the valid zone"""
        raise NotImplementedError()


    def attempt_update(self):
        """
        Using the existing project_optimal_shift, apply this to the new cloud limits and check if another shift needed.

        This is the case where a point cloud is added but is outside of the bounds of the acceptable region.

        min_limits < min_cloud
        max_limits > max_cloud

        TODO: For now, throw error if an object falls outside of these acceptable limits
        """
        raise NotImplementedError()


class A:
    def __init__(self):
        self.opt_g = OptimalShiftManager(self)

if __name__ == "__main__":
    a = A()
    b = A()