### Class Tree Structure

Motivation is to create a base array types that can in turn be used to easily define new classes where constraints that 
often need validation can be handled automatically by the class.

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
DataArray
```


```
PointCloud(FlexibleCoordinates)
├──scalar_fields
└──metadata
```