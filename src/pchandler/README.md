### Class Tree Structure
```
DataArray
├──DataArray1D [M]
│   ├──FilterMask
│   ├──CloudIndexes
│   └──ScalarField
│
├──DataArray2D [MxN]
│   ├──DataArrayNx2
│   │   └──ImageCoordinateSet (B+W Target Detection)
│   │
│   ├──DataArrayNx3
│   │   ├──CoordinateSet3D
│   │   │   ├──Cartesian──┬──FlexibleCoordinates
│   │   │   └──Spherical──┘
│   │   └──ScalarFieldTriplet
│   │   │   ├──Normals
│   │   │   └──RGB*
│   │
│   ├──DepthMap / RangeImage
│   ├──...
│   └──IntensityImage
│
├──DataArray3D  [MxNxP]
│   ├──RGBImage
│   └──ImageStack
│
└──DataArray4D
    ├──RGBImage
    └──RGBImageStack
```

```
PointCloud(FlexibleCoordinates)
├──scalar_fields
└──metadata

```