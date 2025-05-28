import numpy as np
from pydantic import BaseModel, Field, ConfigDict


from pchandler.v2.base_arrays import BaseArray

class TestPydantic:
    def test_initialisation_of_multiple_instances(self):
        a = np.random.rand(10,3)
        b = a + 3
        c = b * 2
        d = c / 4.2

        a = BaseArray(arr=a)
        b = BaseArray(arr=b)
        c = BaseArray(arr=c)
        d = BaseArray(arr=d)

        assert a is not b
        assert b is not c
        assert c is not d

        assert a.arr is not b.arr
        assert b.arr is not c.arr
        assert c.arr is not d.arr

        assert not np.allclose(a, b)
        assert not np.allclose(b, c)
        assert not np.allclose(c, d)

        assert a.model_config == b.model_config
        assert b.model_config == c.model_config
        assert c.model_config == d.model_config

    def test_exclude_on_dump(self):
        class A(BaseArray):
            num: int = Field(default=1)
            name: str|None = Field(default=None, exclude=True)

        a = A(arr=np.random.rand(10,3), name='test')
        assert 'test' == a.name
        a.name = 'abcdef'

        assert 'name' in a.__dict__.keys()

        dumped = a.model_dump()
        assert 'name' not in dumped.keys()

        pydantic_copy = a.model_copy()
        base_array_copy = a.copy()


        assert 'abcdef' == pydantic_copy.name

        # Model_copy by default would return the excluded values
        assert hasattr(pydantic_copy, 'name')
        assert 'abcdef' == pydantic_copy.name
        assert hasattr(base_array_copy, 'name')
        assert getattr(base_array_copy, 'name') is None


    def test_exclude_on_model_copy(self):
        class A(BaseModel):
            num: int = Field(default=1)
            name: str = Field(default='test', exclude=True)

        a = A()

        assert 'name' in a.__dict__.keys()
        assert 'test' in a.__dict__.values()

    # def test_attributes_in_dict(self):
    #     raise NotImplementedError()
    #
    # def test_model_config_overwrite(self):
    #     raise NotImplementedError()
    #
    # def test_cached_properties_excluded(self):
    #     raise NotImplementedError('')