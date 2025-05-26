from .coordinates import CartesianCoordinates, TLSCoordinates
from .scalar_fields import ScalarFieldManager


class BasePointCloud(CartesianCoordinates):
    sfm: ScalarFieldManager

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def sample(self, *index, sub_ok=True):
        self.get_copy(self.sample(*index), update={
            'sfm': self.sfm[*index]
        })

    def reduce(self, *index):
        pass

    def extract(self):
        pass

    def merge(self):
        pass

    def copy(self):
        pass

    def to_o3d(self):
        pass

    @classmethod
    def from_o3d(self, o3d):
        pass

    def to_py4dgeo(self):
        pass

    @classmethod
    def from_py4dgeo(self, py4dgeo):
        pass

    @property
    def normals(self):
        pass

    @property
    def rgb(self):
        pass

    @property
    def intensity(self):
        pass

    @property
    def reflectance(self):
        pass
