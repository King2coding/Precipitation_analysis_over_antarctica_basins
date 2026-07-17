from antarctic_precip_pmb.basins import validate_basin_mapping, validate_region_definitions
from antarctic_precip_pmb.constants import AIS_BASINS, EAIS_BASINS, REGION_BASINS, WAIS_BASINS


def test_basin_mapping_is_valid():
    validate_basin_mapping()


def test_region_definitions_partition_ais():
    validate_region_definitions()
    assert set(WAIS_BASINS).isdisjoint(EAIS_BASINS)
    assert set(WAIS_BASINS) | set(EAIS_BASINS) == set(AIS_BASINS)
    assert tuple(REGION_BASINS["Antarctica"]) == AIS_BASINS
