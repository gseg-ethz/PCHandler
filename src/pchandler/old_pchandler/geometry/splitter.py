import logging
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed, parallel_config

from ..fov import FoV, FoVTree
from .core import PointCloudData
from .filters.spherical_coordinate_filters import FoVFilter

logger = logging.getLogger(__name__.split(".")[0])


class PointCloudSplitter(ABC):
    @abstractmethod
    def split(self):
        pass


class FoVTreePointCloudSplitter(PointCloudSplitter):
    def __init__(self, fov_tree: FoVTree, remove_empty: bool = True, n_jobs: int = -1, method: str = "iterative"):
        """
        Initialize the splitter with configuration options.

        Parameters
        ----------
        remove_empty : bool, default=True
            Whether to skip tasks that result in an empty point cloud.
        n_jobs : int, default=-1
            Number of parallel jobs to use. If -1, use all available cores.
        """
        self.fov_tree = fov_tree
        self.remove_empty = remove_empty
        self.n_jobs = n_jobs
        self.method = method

    def split(self, pcd: PointCloudData) -> dict[str, PointCloudData]:
        """
        Splits a PointCloudData instance using an iterative approach with a FoVTree.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data to be split.

        Returns
        -------
        dict[str, PointCloudData]
            A dictionary mapping FoV identifiers to split PointCloudData objects.
        """
        match self.method:
            case "iterative":
                splits = self._iterative_split(pcd, self.fov_tree)
            case "direct":
                splits = self._direct_split(pcd, self.fov_tree)

        return splits

    def _direct_split(self, pcd: PointCloudData, fov_tree: FoVTree) -> dict[str, PointCloudData]:
        fov_list: list[tuple[str, FoV]] = fov_tree.to_list()
        if len(fov_list) > 1:
            with parallel_config(backend="loky", n_jobs=self.n_jobs, verbose=50, prefer="processes") as config:
                splits = Parallel(return_as="list")(
                    delayed(self._process_direct_task)(pcd, fov_id, fov) for fov_id, fov in fov_list
                )
        else:
            splits = [self._process_direct_task(pcd, fov_list[0][0], fov_list[0][1])]

        pcd.reduce(np.zeros(pcd.nbPoints, dtype=np.bool_))
        return dict(splits)

    def _process_direct_task(self, pcd: PointCloudData, fov_id: str, fov: FoV) -> tuple[str, PointCloudData]:
        new_pcd = FoVFilter(fov).sample(pcd)

        return fov_id, new_pcd

    def _iterative_split(self, pcd: PointCloudData, fov_tree: FoVTree) -> dict[str, PointCloudData]:
        """
        Perform the iterative splitting of the point cloud using a task queue.

        Parameters
        ----------
        pcd : PointCloudData
            The initial point cloud.

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
        """
        Process a single task consisting of a point cloud and a FoVTree node.

        If the node is a leaf, returns the result; otherwise, returns new tasks for each child.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud to process.

        Returns
        -------
        tuple
            A tuple with a dictionary of results (empty if not a leaf) and a list of new tasks.
        """
        if fov_tree.is_leaf():
            return {fov_tree.identifier: pcd}, []

        child_tasks = []
        for child in fov_tree.children.values():
            child_pcd = FoVFilter(child.node).extract(pcd)
            # child_pcd = pcd.extract_angles(child.node)
            if self.remove_empty and child_pcd.nbPoints == 0:
                continue
            child_tasks.append((child_pcd, child))
        return {}, child_tasks


def split_pc_with_fov_tree(
    pcd: PointCloudData, fov_tree: FoVTree, remove_empty: bool = True, n_jobs: int = -1
) -> dict[str, PointCloudData]:
    # -> list[tuple[str, FoV, PointCloudData]]:

    """
    Splits a PointCloudData instance using a FoVTree.

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

    # Setup argumenets for call
    split_packages = [
        (pcd.extract_angles(child.node), child, remove_empty, n_jobs) for child in fov_tree.children.values()
    ]
    if remove_empty:
        split_packages = [sp for sp in split_packages if sp[0].nbPoints]
    # print(*[FoV(**sp[0].fov, unit="rad") for sp in split_packages], sep='\n')

    split = Parallel(n_jobs=n_jobs, prefer="processes", verbose=50, timeout=10 * 60)(
        delayed(split_pc_with_fov_tree)(*split_package) for split_package in split_packages
    )

    split_dict = {k: v for d in split for k, v in d.items()}
    return split_dict
