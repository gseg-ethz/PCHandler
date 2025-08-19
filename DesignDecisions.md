### BaseArray

This class is designed to be an automatic validated, subclassable array like object.
The main idea and perceived functionality is the following:
- Acts much like a Numpy array and has a base array that is used with any other Numpy functionality
> ```python[dracula]
> points = np.random.rand(100, 3)
> intensity = np.random.rand(100,)
> pcd = PointCloudData(points, intensity=intensity)
> ```
- Automatically performs validation on initialisation and attribute assignment to ensure they are of correct type
  - Also shape is validated when defined for Numpy array data

- Can be extended to have additional attributes but fundamental object acts like a numpy array
