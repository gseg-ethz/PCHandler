# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
from GSEGUtils.constants import validate_variables
from joblib import Parallel, cpu_count, delayed, parallel_config
from pydantic import BeforeValidator, Field

from pchandler import PointCloudData
from pchandler.filters import FoVFilter
from pchandler.geometry.spherical import FoV, FoVTree

logger = logging.getLogger(__name__.split(".")[0])


NUM_CPUS = cpu_count()
FoVSplitMethodT = Literal["iterative"] | Literal["direct"]


def check_number_jobs(n_jobs: int):
    """Validate the number of jobs to be started based on the CPU count

    Parameters
    ----------
    n_jobs : int
        Input number of jobs

    Returns
    -------
    int
        Returned number of jobs
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must be -1 or a positive integer value. Zero is not valid.")

    if n_jobs > NUM_CPUS:
        warnings.warn(
            f"Maximum number of jobs entered [{n_jobs}] is greater than the number of cores. "
            f"Using the maximum available CPU count instead = {NUM_CPUS}",
            stacklevel=2,
        )
        return NUM_CPUS

    return n_jobs


NumberJobsT = Annotated[int, Field(gt=-NUM_CPUS, le=NUM_CPUS), BeforeValidator(check_number_jobs)]


class PointCloudSplitter(ABC):
    """Abstract class for point cloud splitting algorithms"""

    @abstractmethod
    def split(self, pcd: PointCloudData) -> dict[str, PointCloudData]:
        """Splits a point cloud into multiple segments.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        dict[str, PointCloudData]
            Split point cloud data. The keys are identifiers for each segment.
        """
        pass


class FoVTreePointCloudSplitter(PointCloudSplitter):
    """Splits a point cloud into smaller subsets based on a field-of-view (FoV) tree"""

    # Todo: Check how validate_variables interacts with initializers...got weird errors
    # DISCUSS: Still relevant?
    @validate_variables
    def __init__(
        self,
        fov_tree: FoVTree,
        remove_empty: bool = True,
        n_jobs: NumberJobsT = -1,
        method: FoVSplitMethodT = "iterative",
    ):
        """Initialize the splitter with configuration options.

        Parameters
        ----------
        fov_tree : FoVTree
            FoV tree structure defining the splits
        remove_empty : bool, default=True
            Skip tasks that result in an empty point cloud.
        n_jobs : int, default=-1
            Number of parallel jobs to use. Defaults to all available cores.
        method : FoVSplitMethodT
            'iterative' or 'direct'
        """
        self.fov_tree = fov_tree
        self.remove_empty = remove_empty
        self.n_jobs = n_jobs
        self.method: FoVSplitMethodT = method

    def split(self, pcd: PointCloudData) -> dict[str, PointCloudData]:
        """Split the point cloud based on the FoVTree.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        dict[str, PointCloudData]
            Split point cloud data. The keys are identifiers for each segment.
        """
        match self.method:
            case "iterative":
                splits = self._iterative_split(pcd, self.fov_tree)
            case "direct":
                splits = self._direct_split(pcd, self.fov_tree)
            case _:
                raise ValueError(f"Invalid method passed for the splitting: {self.method}")

        return splits

    def _direct_split(self, pcd: PointCloudData, fov_tree: FoVTree) -> dict[str, PointCloudData]:
        """Performs the direct split method

        Parameters
        ----------
        pcd : PointCloudData
        fov_tree : FoVTree

        Returns
        -------
        dict[str, PointCloudData]
        """
        fov_list: list[tuple[str, FoV]] = fov_tree.to_list()

        if len(fov_list) > 1:
            with parallel_config(backend="loky", n_jobs=self.n_jobs, verbose=50, prefer="processes"):
                splits = Parallel(return_as="list")(
                    delayed(self._process_direct_task)(pcd, fov_id, fov) for fov_id, fov in fov_list
                )
        else:
            splits = [self._process_direct_task(pcd, fov_list[0][0], fov_list[0][1])]

        pcd.reduce(np.zeros(len(pcd), dtype=np.bool_))
        return dict(splits)

    def _process_direct_task(self, pcd: PointCloudData, fov_id: str, fov: FoV) -> tuple[str, PointCloudData]:
        """Process a single task consisting

        Parameters
        ----------
        pcd : PointCloudData
        fov_id : str
        fov : FoV

        Returns
        -------
        tuple[str, PointCloudData]
            Identifier and split point cloud data.
        """
        new_pcd = FoVFilter(fov).sample(pcd)

        return fov_id, new_pcd

    def _iterative_split(self, pcd: PointCloudData, fov_tree: FoVTree) -> dict[str, PointCloudData]:
        """Perform the iterative split method

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        dict[str, PointCloudData]
            A dictionary of split point clouds.
        """
        results = {}
        tasks = [(pcd, fov_tree)]

        while tasks:
            if len(tasks) > 1:
                # Process tasks at the current level concurrently
                with parallel_config(backend="loky", n_jobs=self.n_jobs, verbose=50, prefer="processes"):
                    level_results = Parallel(return_as="list")(
                        delayed(self._process_iterative_task)(task_pcd, task_fov) for task_pcd, task_fov in tasks
                    )
            else:
                # Process sequentially when only one task is present
                level_results = [self._process_iterative_task(tasks[0][0], tasks[0][1])]

            new_tasks = []
            for res, child_tasks in level_results:
                results.update(res)
                new_tasks.extend(child_tasks)
            tasks = new_tasks

        return results

    def _process_iterative_task(
        self, pcd: PointCloudData, fov_tree: FoVTree
    ) -> tuple[dict[str, PointCloudData], list[tuple[PointCloudData, FoVTree]]]:
        """Process a single iteration task

        If the node is a leaf, returns the result; otherwise, returns new tasks for each child.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        tuple
            A tuple with a dictionary of results (empty if not a leaf) and a list of new tasks.
        """
        if fov_tree.is_leaf():
            return {fov_tree.identifier: pcd}, []

        child_tasks: list[tuple[PointCloudData, FoVTree]] = []
        if fov_tree.children is not None:
            for child in fov_tree.children.values():
                child_pcd = FoVFilter(child.node).extract(pcd)
                # child_pcd = pcd.extract_angles(child.node)
                if self.remove_empty and len(child_pcd) == 0:
                    continue
                child_tasks.append((child_pcd, child))
        return dict(), child_tasks


def split_pc_with_fov_tree(
    pcd: PointCloudData, fov_tree: FoVTree, remove_empty: bool = True, n_jobs: NumberJobsT = -1
) -> dict[str, PointCloudData]:
    # -> list[tuple[str, FoV, PointCloudData]]:
    """Splits a PointCloudData instance using a FoVTree.

    Parameters
    ----------
    pcd : PointCloudData
        The point cloud data to be split.
    fov_tree : FoVTree
        A tree structure defining the field of view regions for splitting the point cloud.
    remove_empty : bool, default=True
        Whether to remove empty splits (i.e., regions with no points).
    n_jobs : int, default=-1
        The number of parallel jobs to use. If -1, all available cores are used.

    Returns
    -------
    dict[str, PointCloudData]
        A dictionary where keys are FoV identifiers and values are the split PointCloudData objects.
    """
    # if fov_tree.is_leaf() or pcd.nbPoints <= self.minimum_nb_points:
    #     return [(fov_tree.identifier, fov_tree.node, pcd,)]
    if fov_tree.is_leaf():
        return {fov_tree.identifier: pcd}
        # return [(fov_tree.identifier, fov_tree.node, pcd,)]

    # Setup arguments for call
    if fov_tree.children is not None:
        split_packages = [
            (FoVFilter(child.node).extract(pcd), child, remove_empty, n_jobs) for child in fov_tree.children.values()
        ]
    else:
        split_packages = []

    if remove_empty:
        split_packages = [sp for sp in split_packages if len(sp[0])]
    # print(*[FoV(**sp[0].fov, unit="rad") for sp in split_packages], sep='\n')

    split = Parallel(n_jobs=n_jobs, prefer="processes", verbose=50, timeout=10 * 60)(
        delayed(split_pc_with_fov_tree)(*split_package) for split_package in split_packages
    )

    split_dict = {k: v for d in split for k, v in d.items()}
    return split_dict
