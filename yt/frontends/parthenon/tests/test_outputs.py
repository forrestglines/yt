import numpy as np
from numpy.testing import assert_allclose, assert_equal

from yt.frontends.parthenon.api import ParthenonDataset
from yt.loaders import load
from yt.testing import (
    assert_true,
    requires_file,
    units_override_check,
)
from yt.utilities.answer_testing.framework import (
    GenericArrayTest,
    data_dir_load,
    requires_ds,
)

_fields_advection2d = (
    ("parthenon", "advected_0_0"),
    ("parthenon", "one_minus_advected"),
    ("parthenon", "one_minus_advected_sq"),
    ("parthenon", "one_minus_sqrt_one_minus_advected_sq_12"),
    ("parthenon", "one_minus_sqrt_one_minus_advected_sq_37"),
)

# Simple 2D test (advected spherical blob) with AMR from the main Parthenon test suite
# adjusted so that x1 != x2.
# Ran with `./example/advection/advection-example -i ../tst/regression/test_suites/output_hdf5/parthinput.advection parthenon/mesh/nx1=128 parthenon/mesh/x1min=-1.0 parthenon/mesh/x1max=1.0 Advection/vx=2`
# on changeset e5059ad
advection2d = "advection_2d.out0.final.phdf"

@requires_ds(advection2d)
def test_loading_data():
    ds = data_dir_load(advection2d)
    assert_equal(str(ds), "advection_2d.out0.final")
    dd = ds.all_data()
    # test mesh dims
    vol = np.product(ds.domain_right_edge - ds.domain_left_edge)
    assert_equal(vol, ds.quan(2.0, "code_length**3"))
    assert_allclose(dd.quantities.total_quantity("cell_volume"), vol)
    # test data
    for field in _fields_advection2d:

        def field_func(name):
            return dd[name]

        yield GenericArrayTest(ds, field_func, args=[field])

    # reading data of two fields and compare against each other (data is squared in output)
    ad = ds.all_data()
    assert_allclose(
        ad[("parthenon", "one_minus_advected")] ** 2.0,
        ad[("parthenon", "one_minus_advected_sq")],
    )

    # check if the peak is in the domain center (and at the highest refinement level)
    dist_of_max_from_center = np.linalg.norm(
        ad.quantities.max_location(("parthenon", "advected_0_0"))[1:] - ds.domain_center
    )

    dx_min, dx_max = ad.quantities.extrema(("index", "dx"))
    dy_min, dy_max = ad.quantities.extrema(("index", "dy"))

    assert_true(dist_of_max_from_center < np.min((dx_min, dy_min)))

athenapk_disk = "athenapk.prim.disk.phdf"

@requires_ds(athenapk_disk)
def test_AthenaPKDataset():
    assert isinstance(data_dir_load(athenapk_disk), ParthenonDataset)

@requires_ds(athenapk_disk)
def test_load_cylindrical():
    ds = data_dir_load(athenapk_disk)

    assert_equal(ds.domain_left_edge.in_units("code_length").v[:2],(0.5,0))
    assert_equal(ds.domain_right_edge.in_units("code_length").v[:2],(2.0,2*np.pi))

@requires_ds(athenapk_disk)
def test_units():
    ds = data_dir_load(athenapk_disk)
    assert_allclose(float(ds.quan(1,"code_time"  ).in_units("Gyr" )),1   ,rtol=1e-8)
    assert_allclose(float(ds.quan(1,"code_length").in_units("Mpc" )),1   ,rtol=1e-8)
    assert_allclose(float(ds.quan(1,"code_mass"  ).in_units("msun")),1e14,rtol=1e-8)

_fields_derived = (
    ("gas", "temperature"),
)

@requires_ds(athenapk_disk)
def test_derived_fields():
    ds = data_dir_load(athenapk_disk)
    dd = ds.all_data()

    # test data
    for field in _fields_derived:

        def field_func(name):
            return dd[name]

        yield GenericArrayTest(ds, field_func, args=[field])

