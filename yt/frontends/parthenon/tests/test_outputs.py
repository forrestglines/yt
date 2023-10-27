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
    small_patch_amr,
)

_fields_advection2d = (
    ("parthenon", "Advected_0_0"),
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
def test_disk():
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
        ad.quantities.max_location(("parthenon", "Advected_0_0"))[1:] - ds.domain_center
    )

    dx_min, dx_max = ad.quantities.extrema(("index", "dx"))
    dy_min, dy_max = ad.quantities.extrema(("index", "dy"))

    assert_true(dist_of_max_from_center < np.min((dx_min, dy_min)))


_fields_AM06 = ("temperature", "density", "velocity_magnitude", "magnetic_field_x")

AM06 = "AM06/AM06.out1.00400.athdf"


@requires_ds(AM06)
def test_AM06():
    ds = data_dir_load(AM06)
    assert_equal(str(ds), "AM06.out1.00400")
    for test in small_patch_amr(ds, _fields_AM06):
        test_AM06.__name__ = test.description
        yield test


uo_AM06 = {
    "length_unit": (1.0, "kpc"),
    "mass_unit": (1.0, "Msun"),
    "time_unit": (1.0, "Myr"),
}


@requires_file(AM06)
def test_AM06_override():
    # verify that overriding units causes derived unit values to be updated.
    # see issue #1259
    ds = load(AM06, units_override=uo_AM06)
    assert_equal(float(ds.magnetic_unit.in_units("gauss")), 9.01735778342523e-08)


@requires_file(AM06)
def test_units_override():
    units_override_check(AM06)


@requires_file(AM06)
def test_AthenaPPDataset():
    assert isinstance(data_dir_load(AM06), ParthenonDataset)
