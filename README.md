# Motivation / Justification


## Custom Numpy Array Class

Motivations
- Object Oriented with high level polymorphism

```
Class CoordinateSet3D(CustomArray):
    __num_cols = 3

    def __len__(self):
        return self.shape[0]

class SphericalCoordinates(CoordinateSet):
    @property
    def hz(self) -> np.ndarray: ...

    @property
    def v(self) -> np.ndarray: ...

    @property
    def r(self) -> np.ndarray: ...

    def to_cartesian(self) -> np.ndarray: ...
```
- Treated like a numpy array directly for array / matrix operations
  - `transformed_coords = R @ CustomArray + T`


- Ability to append extra attributes to the main array like a dataclass definition
```python
array.info = "First Data Set"
array.metadata = {transform: array}
array.transforms.history.append(('Image plane projection', array[3, 3]))
```

- Strict type enforcement to mitigate errors from user input or during development
```python
class ScalarField(CustomArray):
    label = ValidatedField(data=data,
                           name=name,
                           type=type,
                           validation_func=None,
                           dtype=None)

    arr = ValidatedField(array, 'arr', np.ndarray, check_all_positive, dtype=np.float)

```
- Automatically perform attribute validation on setting
```python
class ValidatedField:
    def __init__(self, data, name, type,
                 validation_funcs: list[Callable] = None,
                 dtype: np.ndarray.dtype = None): ...
        self.validation_func = validation_func

    def validate(value):
        if self.validation_funcs is not None:
            for func in self.validation_funcs:
                array = validation_func(array)

    def __setattr__(self, name, value
        value = self.validate(value)
        super().__setattr__(self, name, value)
```
- Full control of the degree of immutability of the array (dataclass only limited attribute setting)

```python
class ImmutabilityLevels(Enum):
    FULL = 0
    ATTRIBUTE_ONLY = 1
    NONE = 2

array = CustomArray(np.random.rand(3,3))
array.setImmutability(ImmutabilityLevels.NONE)
array = np.random.rand(3,3)
# No Error

array.setImmutability(ImmutabilityLevels.FULL)
array = np.random.rand(3,3)
# Raises attribute error
array[1, 1] = 99
# Raises writing error

array.setImmutability(ImmutabilityLevels.ATTRIBUTE_ONLY)
array = np.random.rand(3,3)
# Raises attribute error

array[:, 2] = np.ones(3)
# NO ERROR
```

- Other Key Considerations
  - DRY - Don't Repeat yourself
  - Code readability
  - Self descriptive code - few comments required
  - Protocol based Duck Typing

- The goal is that it should be easily extendable to adapt to other situations
  - So avoiding spaghetti code and other dependencies
- Other analysis or algorithm functions should be setup in a strategy design pattern for easier data processing workflows
