from __future__ import annotations

from typing import Any, Mapping, Optional, Self, Annotated

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator, BeforeValidator

from ..base_types import Array_4x4_T, Array_Nx3_T, Vector_3_T
from ..validators import extract_array
from .coordinates import CartesianCoordinates
from .optimal_shift import OptimizedShift

from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import NormalFields, RGBFields, ScalarField
from .transforms import Transform, TransformLedger

# TODO check for a better converter - TypeAdapter?

class PointCloudData(CartesianCoordinates):
    arr: Array_Nx3_T = Field(alias="xyz")
    transform_ledger: Annotated[
        TransformLedger[str, [Transform]],
        Field(default_factory=TransformLedger),
        BeforeValidator(lambda value: value if not isinstance(value, TransformLedger) else TransformLedger(**value))
    ]
    scalar_fields: ScalarFieldManager | dict[str, ScalarField] = Field(default_factory=ScalarFieldManager)
    optimized_shift: Optional[OptimizedShift]

    def __init__(
        self,
        xyz: npt.NDArray[np.floating] | Array_Nx3_T | CartesianCoordinates,
        *,
        rgb: Optional[npt.NDArray[Any, np.uint8] | RGBFields] = None,
        normals: Optional[npt.NDArray[Any, np.float32] | NormalFields] = None,
        intensity: Optional[npt.NDArray | ScalarField] = None,
        reflectance: Optional[npt.NDArray | ScalarField] = None,
        optimized_shift: OptimizedShift | ellipsis | None = Ellipsis,
        socs_origin: Optional[np.ndarray] = None,
        scalar_fields: Optional[ScalarFieldManager | dict] = None,
        project_transformation: Optional[Array_4x4_T] = None,
        transform_ledger: Optional[TransformLedger] = None,
        frozen: bool = False
    ):
        if scalar_fields is None:
            scalar_fields = {}

        scalar_fields = ScalarFieldManager(None, fields=scalar_fields)

        if isinstance(rgb, np.ndarray):
            rgb = RGBFields(rgb)

        if isinstance(normals, np.ndarray):
            normals = NormalFields(normals)

        if isinstance(intensity, np.ndarray):
            intensity = ScalarField(intensity, name="intensity")

        if isinstance(reflectance, np.ndarray):
            reflectance = ScalarField(reflectance, name="reflectance")

        for field in (rgb, normals, intensity, reflectance):
            if field is not None:
                scalar_fields.add_field(field)

        # TODO implement post v2.0
        # if transform_ledger is not None:
        #     kwargs['transform_ledger'] = TransformLedger()

        if optimized_shift is Ellipsis:
            optimized_shift = OptimizedShift(np.zeros(3, dtype=np.float32))

        if optimized_shift is not None:
            optimized_shift = optimized_shift.register(self, xyz)

        xyz = xyz - optimized_shift.optimal_shift if optimized_shift is not None else xyz

        # TODO Add an easy accessor to the original point cloud data (e.g. at_socs)

        # TODO Propagate this through to scalar_fields (ScalarField should set this if the parent has it)
        self.model_config["frozen"] = frozen

        super().__init__(
            xyz=xyz,
            scalar_fields = scalar_fields,
            optimized_shift = optimized_shift,
            socs_origin = socs_origin,
            project_transformation = project_transformation,
            transform_ledger = transform_ledger if transform_ledger is not None else TransformLedger(),
        )


    def __hash__(self) -> int:
        return id(self)


    def update_shift(self, delta_shift: Vector_3_T):
        self.xyz = self.xyz + delta_shift


    @model_validator(mode="after")
    def update_parent_weakref(self) -> Self:
        """Revalidate model to ensure that the weakref points to the correct object"""
        if isinstance(self.scalar_fields, ScalarFieldManager):
            self.scalar_fields.parent = self

        elif isinstance(self.scalar_fields, dict):
            self.scalar_fields = ScalarFieldManager(parent=self, fields=self.scalar_fields)

        elif self.scalar_fields is None:
            self.scalar_fields = ScalarFieldManager(parent=self)

        return self

    @property
    def normals(self):
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: np.ndarray|NormalFields):
        self.scalar_fields.normals = value

    @property
    def rgb(self):
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: np.ndarray|RGBFields):
        self.scalar_fields.rgb = value

    @property
    def intensity(self):
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: np.ndarray|ScalarField):
        self.scalar_fields.intensity = value

    @property
    def reflectance(self):
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: np.ndarray|ScalarField):
        self.scalar_fields.reflectance = value

    def __setitem__(self, key, value: PointCloudData):
        raise IndexError(
            f"Setting items in PointCloudData is not supported. Consider using the copy or "
            f"dump data to a dict and reinstantiate."
        )


    def copy(self,
             array: npt.NDArray | Self | None = None,
             *,
             update: Mapping[str, Any] = None,
             **kwargs) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """

        update = update or {}

        # array is passed when sampling and advanced indexing automatically makes a copy
        if array is not None:
            if isinstance(array, CartesianCoordinates):
                update["xyz"] = array.arr
            elif isinstance(array, np.ndarray):
                update["xyz"] = array
            else:
                raise TypeError(f"Invalid type of array passed: {type(array)}. Should be PointCloudData or np.ndarray")

        # Create a copy of the rest of the fields
        update = self.model_dump(exclude=(set(update.keys()))) | update

        return type(self)(update.pop('xyz'), **update)

    def sample(self, mask) -> PointCloudData:
        mask = self.create_mask(mask)
        return self.copy(self.arr[mask, :], update={"scalar_fields": self.scalar_fields.sample(mask)})

    def reduce(self, mask):
        super().reduce(mask)
        self.scalar_fields.reduce(mask)

    def extract(self, mask):
        extracted = super().extract(mask)
        return extracted

    @staticmethod
    def merge(*pcds: PointCloudData):
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        if not all([pcds[0].optimized == pcd.optimized for pcd in pcds[1:]]):
            raise ValueError('Can only merge point clouds if they are all optimized or unoptimized.')

        if isinstance(pcds[0].socs_origin, np.ndarray):
            if not all([np.all(pcds[0].socs_origin == pcd.socs_origin) for pcd in pcds[1:]]):
                raise ValueError("Cannot merge point clouds where some origins are known and some are ambiguous")

        for pcd in pcds:
            if pcd.socs_origin is not None:
                raise ValueError("Cannot merge point clouds where some origins are known and some are ambiguous")

        # TODO update when implementing the transformations
        if pcds[0].project_transformation is None:
            for pcd in pcds[1:]:
                if pcd.project_transformation is not None:
                    raise ValueError("Cannot merge point clouds where only some project transforms are defined")
        else:
            for pcd in pcds[1:]:
                if not isinstance(pcd.project_transformation, np.ndarray):
                    raise ValueError("Cannot merge point clouds where only some project transforms are defined")

        xyz = np.concatenate([pcd.xyz for pcd in pcds], axis=0)

        return PointCloudData(xyz, scalar_fields=scalar_fields)

    def to_o3d(self):
        """
            Converts the point cloud to an Open3D `PointCloud` object.

            Returns
            -------
            o3d.geometry.PointCloud
                An Open3D representation of the point cloud.
            pcd_o3d = o3d.geometry.PointCloud()
            if self.global_coordinate_shift is None:
                pcd_o3d.points = o3d.utility.Vector3dVector(self.xyz)
            else:
                pcd_o3d.points = o3d.utility.Vector3dVector((self.xyz + self.global_coordinate_shift).astype(np.float64))
            return pcd_o3d
        """
        raise NotImplementedError



    @classmethod
    def from_o3d(cls, o3d):
        """
            @classmethod
            def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud, scan_center: Optional[NDArray[np.float_]] = None) -> Self:
                Creates a `PointCloudData` instance from an Open3D `PointCloud`.

                Parameters
                ----------
                pcd_o3d : o3d.geometry.PointCloud
                    An Open3D `PointCloud` object.
                scan_center : np.ndarray, optional
                    The scan center for spherical coordinate calculations.

                Returns
                -------
                PointCloudData
                    A new instance of the `PointCloudData` class.
                return cls(np.asarray(pcd_o3d.points), spherical_coordinates_origin=scan_center)
        """
        raise NotImplementedError

    # # DECIDE Implement in PCHandler or in pc2image
    # @classmethod
    # def from_range_image(
    #         cls,
    #         range_data: NDArray[np.floating],
    #         fov: FoV,
    #         scalar_fields: Optional[dict[str, NDArray[np.generic]] | ScalarFieldManager] = None,
    #         spherical_coordinates_origin: Optional[NDArray[np.float_]] = None,
    # ) -> Self:
    #     """
    #     Creates a `PointCloudData` instance from a range image.
    #
    #     Parameters
    #     ----------
    #     range_data : NDArray[np.floating]
    #         A 2D array representing the range values.
    #     fov : FoV
    #         The field of view defining the angular limits of the range image.
    #     scalar_fields : dict[str, NDArray[np.generic]] | ScalarFieldManager, optional
    #         Scalar fields corresponding to the range data.
    #     spherical_coordinates_origin : NDArray[np.float_], optional
    #         The origin for spherical coordinate calculations.
    #
    #     Returns
    #     -------
    #     PointCloudData
    #         A new instance of the `PointCloudData` class.
    #     """
    #     sfm = ScalarFieldManager() if scalar_fields is None else scalar_fields
    #     if not isinstance(sfm, ScalarFieldManager) and scalar_fields is not None:
    #         sfm = ScalarFieldManager()
    #         for sf_id, sf in scalar_fields.items():
    #             sfm.create_field(sf_id, sf.flatten())
    #
    #     resolution = range_data.shape
    #     elevation_range = np.linspace(
    #         fov.elevation_min, fov.elevation_max, num=resolution[0], endpoint=True, dtype=np.float32
    #     )
    #     horizontal_range = np.linspace(
    #         fov.horizontal_min, fov.horizontal_max, num=resolution[1], endpoint=True, dtype=np.float32
    #     )
    #
    #     elevation_mesh, horizontal_mesh = np.meshgrid(elevation_range, horizontal_range, indexing="ij")
    #
    #     ranges = range_data.flatten()
    #     elevations = elevation_mesh.flatten()
    #     horizontals = horizontal_mesh.flatten()
    #
    #     spherical_coordinates = np.vstack((ranges, elevations, horizontals)).T
    #     spherical_coordinates = spherical_coordinates[~np.isnan(ranges), :]
    #
    #     sfm_reduced = sfm[~np.isnan(ranges)]
    #
    #     return cls.from_spherical_coordinates(spherical_coordinates, sfm_reduced, spherical_coordinates_origin)