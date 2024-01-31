import pchandler as pch
from pchandler.util import AngleUnit

my_fov1 = pch.fov.FoV(horizontal_min=0, horizontal_max=400, elevation_min=0, elevation_max=180, unit="gon")
my_fov2 = pch.fov.FoV(horizontal_min=0, horizontal_max=400, elevation_min=0, elevation_max=180, unit=AngleUnit.GON)

print(1)
