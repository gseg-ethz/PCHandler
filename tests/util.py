import numpy as np

def test_np_operator_mixins(self):
    a = self.cls(arr=np.ones((100, 3)))

    # Subtract
    a = a - 1.0
    assert np.all(np.isclose(a, 0))
    assert isinstance(a, (self.cls, np.ndarray))  # change after assignment

    # Add
    a = a + 2.0
    assert np.all(np.isclose(a, 2))
    assert isinstance(a, (self.cls, np.ndarray))  # change after assignment

    # Divide
    a = a / 0.5
    assert np.all(np.isclose(a, 4))
    assert isinstance(a, (self.cls, np.ndarray))  # change after assignment

    # Multiply
    a = a * 3
    assert np.all(np.isclose(a, 12))
    assert isinstance(a, (self.cls, np.ndarray))  # change after assignment
