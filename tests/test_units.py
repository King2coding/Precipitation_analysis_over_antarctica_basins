import numpy as np

from antarctic_precip_pmb.units import gt_to_mm_water_equivalent, mm_water_equivalent_to_gt


def test_gt_mm_roundtrip():
    area_m2 = np.array([1.0e12, 2.0e12])
    gt = np.array([1.0, 4.0])
    mm = gt_to_mm_water_equivalent(gt, area_m2)
    assert np.allclose(mm, [1.0, 2.0])
    assert np.allclose(mm_water_equivalent_to_gt(mm, area_m2), gt)
