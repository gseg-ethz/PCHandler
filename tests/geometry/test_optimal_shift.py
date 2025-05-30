import weakref

import pytest
import numpy as np

from pchandler.v2.geometry.optimal_shift import OSM_Manager
from pchandler.v2.geometry.core_pydantic import PointCloudData


def test_register():
    a = PointCloudData(xyz = np.random.rand(100, 3) * 100, optimal=True)
    assert any(id(wref) == id(a) for wref in OSM_Manager.point_clouds)
    OSM_Manager.reset()

def test_weakref_list():
    pcd_list = list()
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 100, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 200, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 400, optimal=True))

    assert len(OSM_Manager.point_clouds) == 3

    for i, pcd in enumerate(pcd_list):
        assert any(id(wref) == id(pcd) for wref in OSM_Manager.point_clouds)  # ensure pcd is in the set
        assert np.allclose(pcd.min(), pcd_list[i].min())
        assert np.allclose(pcd.max(), pcd_list[i].max())

    del_id = id(pcd_list[0])
    del pcd_list[0]

    assert not any(id(wref) == del_id for wref in OSM_Manager.point_clouds)
    assert len(OSM_Manager.point_clouds) == 2
    OSM_Manager.reset()

def test_optimal_shift_max_size():
    pcd_list = list()
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*100, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*200, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*400, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*800, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*1000, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*2000, optimal=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100,3)*5000, optimal=True))

    assert len(OSM_Manager.point_clouds) == 7

    with pytest.raises(ValueError):
        PointCloudData(xyz=np.random.rand(100,3)*20000, optimal=True)

    OSM_Manager.reset()

def test_optimal_shift_reset():
    a = PointCloudData(xyz=np.random.rand(100,3)*100, optimal=True)
    b = PointCloudData(xyz=np.random.rand(100,3)*200, optimal=True)
    c = PointCloudData(xyz=np.random.rand(100,3)*400, optimal=True)
    d = PointCloudData(xyz=np.random.rand(100,3)*800, optimal=True)
    assert len(OSM_Manager.point_clouds) == 4
    OSM_Manager.reset()

    assert len(OSM_Manager.point_clouds) == 0
    assert OSM_Manager.current_bbox is None
    assert OSM_Manager.bounding_boxes is None
    assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
    assert isinstance(OSM_Manager.point_clouds, weakref.WeakSet)

def test_adapting_optimal_shift():
    # Test that on each addition, the new optimal shift center is calculated.
    # When it reaches max size, throws error
    a_origin =  np.array([100, -400, 100], dtype=np.float32)
    a_ = (np.random.rand(10,3)*400).astype(np.float32) + a_origin
    b_origin = np.array([3400, -1200, -100], dtype=np.float32)
    b_ = (np.random.rand(10,3)*400).astype(np.float32) + b_origin
    c_origin = np.array([-4000, -6500, -7000], dtype=np.float32)
    c_ = (np.random.rand(10,3)*400).astype(np.float32) + c_origin
    d_origin = np.array([-6350, 5320, -8200], dtype=np.float32)
    d_ = (np.random.rand(10,3)*400).astype(np.float32) + d_origin
    e_ = (np.random.rand(10,3)*400).astype(np.float32) - np.array([20000]).astype(np.float32)

    # Initialise the first, opt shift is too close to origin (< 500)
    a = PointCloudData(xyz=a_.copy(), socs_origin=a_origin.copy(), optimal=True)
    assert OSM_Manager.optimal_shift.shape == (3,)
    assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
    last_shift = OSM_Manager.optimal_shift.copy()
    last_a_socs = a.socs_origin.copy()

    # Initialise 2nd
    b = PointCloudData(xyz=b_.copy(), socs_origin=b_origin.copy(), optimal=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)
    last_a_socs = a.socs_origin.copy()
    last_shift = OSM_Manager.optimal_shift.copy()

    # Initialise 3rd
    c = PointCloudData(xyz=c_.copy(), socs_origin=c_origin.copy(), optimal=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)
    last_a_socs = a.socs_origin.copy()
    last_shift = OSM_Manager.optimal_shift.copy()

    # Initialise 4th
    d = PointCloudData(xyz=d_.copy(), socs_origin=d_origin.copy(), optimal=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)


    with pytest.raises(ValueError):
        PointCloudData(xyz=e_.copy().astype(np.float32), optimal=True)

    # check that all the socs are different
    assert not np.allclose(a_origin, a.socs_origin)
    assert not np.allclose(b_origin, b.socs_origin)
    assert not np.allclose(c_origin, c.socs_origin)
    assert not np.allclose(d_origin, d.socs_origin)
    assert np.allclose(a_, a + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(b_, b + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(c_, c + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(d_, d + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(a_origin, a.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(b_origin, b.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(c_origin, c.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(d_origin, d.socs_origin + OSM_Manager.optimal_shift)

    OSM_Manager.reset()