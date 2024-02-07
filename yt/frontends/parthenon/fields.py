import numpy as np

from yt._typing import KnownFieldsT
from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.physical_constants import kboltz, mh

mag_units = "code_magnetic"
pres_units = "code_mass/(code_length*code_time**2)"
rho_units = "code_mass / code_length**3"
vel_units = "code_length / code_time"
mom_units = "code_mass / code_length**2 / code_time"
eng_units = "code_mass / code_length / code_time**2"


def velocity_field(mom_field):
    def _velocity(field, data):
        return data[mom_field] / data["gas", "density"]

    return _velocity


def _cooling_time_field(field, data):
    cooling_time = (
        data["gas", "density"]
        * data["gas", "specific_thermal_energy"]
        / data["gas", "cooling_rate"]
    )

    # Set cooling time where Cooling_Rate==0 to infinity
    inf_ct_mask = data["cooling_rate"] == 0
    cooling_time[inf_ct_mask] = data.ds.quan(np.inf, "s")

    return cooling_time


class ParthenonFieldInfo(FieldInfoContainer):
    known_other_fields: KnownFieldsT = (
        # Need to provide info for both primitive and conserved variable as they
        # can be written indepdendently (or even in the same output file).
        # New field naming (i.e., "variable_component") of primitive variables
        ("prim_density", (rho_units, ["density"], None)),
        ("prim_velocity_1", (vel_units, ["velocity_x"], None)),
        ("prim_velocity_2", (vel_units, ["velocity_y"], None)),
        ("prim_velocity_3", (vel_units, ["velocity_z"], None)),
        ("prim_pressure", (pres_units, ["pressure"], None)),
        # Magnetic fields carry units of 1/sqrt(pi) so we cannot directly forward
        # and need to setup aliases below.
        ("prim_magnetic_field_1", (mag_units, [], None)),
        ("prim_magnetic_field_2", (mag_units, [], None)),
        ("prim_magnetic_field_3", (mag_units, [], None)),
        # New field naming (i.e., "variable_component") of conserved variables
        ("cons_density", (rho_units, ["density"], None)),
        ("cons_momentum_density_1", (mom_units, ["momentum_density_x"], None)),
        ("cons_momentum_density_2", (mom_units, ["momentum_density_y"], None)),
        ("cons_momentum_density_3", (mom_units, ["momentum_density_z"], None)),
        ("cons_total_energy_density", (eng_units, ["total_energy_density"], None)),
        # Magnetic fields carry units of 1/sqrt(pi) so we cannot directly forward
        # and need to setup aliases below.
        ("cons_magnetic_field_1", (mag_units, [], None)),
        ("cons_magnetic_field_2", (mag_units, [], None)),
        ("cons_magnetic_field_3", (mag_units, [], None)),
        # Legacy naming. Given that there's no conflict with the names above,
        # we can just define those here so that the frontend works with older data.
        ("Density", (rho_units, ["density"], None)),
        ("Velocity1", (mom_units, ["velocity_x"], None)),
        ("Velocity2", (mom_units, ["velocity_y"], None)),
        ("Velocity3", (mom_units, ["velocity_z"], None)),
        ("Pressure", (pres_units, ["pressure"], None)),
        ("MagneticField1", (mag_units, [], None)),
        ("MagneticField2", (mag_units, [], None)),
        ("MagneticField3", (mag_units, [], None)),
        ("MomentumDensity1", (mom_units, ["momentum_density_x"], None)),
        ("MomentumDensity2", (mom_units, ["momentum_density_y"], None)),
        ("MomentumDensity3", (mom_units, ["momentum_density_z"], None)),
        ("TotalEnergyDensity", (eng_units, ["total_energy_density"], None)),
    )

    def setup_fluid_fields(self):
        from yt.fields.magnetic_field import setup_magnetic_field_aliases

        unit_system = self.ds.unit_system
        # Add velocity fields (if only momemtum densities are given)
        for i, comp in enumerate(self.ds.coordinates.axis_order):
            # Support both current and legacy scheme
            for mom_field_name in ["MomentumDensity", "cons_momentum_density_"]:
                mom_field = ("parthenon", f"{mom_field_name}{i+1}")
                if mom_field in self.field_list:
                    self.add_field(
                        ("gas", f"velocity_{comp}"),
                        sampling_type="cell",
                        function=velocity_field(mom_field),
                        units=unit_system["velocity"],
                    )
        # Figure out thermal energy field
        if ("parthenon", "Pressure") in self.field_list or (
            "parthenon",
            "prim_pressure",
        ) in self.field_list:

            def _specific_thermal_energy(field, data):
                # TODO This only accounts for ideal gases with adiabatic indices
                return (
                    data["gas", "pressure"]
                    / (data.ds.gamma - 1.0)
                    / data["gas", "density"]
                )

            self.add_field(
                ("gas", "specific_thermal_energy"),
                sampling_type="cell",
                function=_specific_thermal_energy,
                units=unit_system["specific_energy"],
            )
        elif ("parthenon", "TotalEnergyDensity") in self.field_list or (
            "parthenon",
            "cons_total_energy_density",
        ) in self.field_list:

            def _specific_thermal_energy(field, data):
                eint = (
                    data["gas", "total_energy_density"]
                    - data["gas", "kinetic_energy_density"]
                )

                if (
                    ("parthenon", "MagneticField1") in self.field_list
                    or ("parthenon", "prim_magnetic_field_1") in self.field_list
                    or ("parthenon", "cons_magnetic_field_1") in self.field_list
                ):
                    eint -= data["gas", "magnetic_energy_density"]
                return eint / data["gas", "density"]

            self.add_field(
                ("gas", "specific_thermal_energy"),
                sampling_type="cell",
                function=_specific_thermal_energy,
                units=unit_system["specific_energy"],
            )

        # Add temperature field
        def _temperature(field, data):
            return (
                (data["gas", "pressure"] / data["gas", "density"])
                * data.ds.mu
                * mh
                / kboltz
            )

        self.add_field(
            ("gas", "temperature"),
            sampling_type="cell",
            function=_temperature,
            units=unit_system["temperature"],
        )

        # We can simply all all variants as only fields present will be added
        setup_magnetic_field_aliases(
            self, "parthenon", ["MagneticField%d" % ax for ax in (1, 2, 3)]
        )
        setup_magnetic_field_aliases(
            self, "parthenon", ["prim_magnetic_field_%d" % ax for ax in (1, 2, 3)]
        )
        setup_magnetic_field_aliases(
            self, "parthenon", ["cons_magnetic_field_%d" % ax for ax in (1, 2, 3)]
        )

        #This section of code was broken a while ago
        #if (
        #    "cooling_table_filename" in self.ds.parameters
        #    and "He_mass_fraction" in self.ds.parameters
        #):
        #    # A cooling table is provided - compute "Cooling_Rate" and "Cooling_Time"

        #    cooling_table = np.loadtxt(self.ds.parameters["cooling_table_filename"])
        #    log_temps = cooling_table[
        #        :, self.ds.parameters["cooling_table_log_temp_col"]
        #    ]
        #    log_lambdas = cooling_table[
        #        :, self.ds.parameters["cooling_table_log_lambda_col"]
        #    ]

        #    lambdas_units = self.ds.quan(
        #        self.ds.parameters["cooling_table_lambda_units_cgs"],
        #        "erg*cm**3/s",
        #    )

        #    def _cooling_rate_field(field, data):
        #        nonlocal log_temps, log_lambdas, lambdas_units
        #        log_temp = np.log10(data["gas", "temperature"].in_units("K").v)
        #        log_lambda = np.interp(log_temp, log_temps, log_lambdas)

        #        # Zero cooling below the table
        #        log_lambda[log_temp < log_temps[0]] = 0

        #        # Interpolate free-free cooling (~T^1/2) above the table
        #        log_lambda[log_temp > log_temps[-1]] = (
        #            0.5 * log_temp[log_temp > log_temps[-1]]
        #            - 0.5 * log_temps[-1]
        #            + log_lambdas[-1]
        #        )

        #        lambda_ = 10 ** (log_lambda) * lambdas_units

        #        # Currently assuming a fully ionized gas without metallicity.
        #        # This needs to be updated when AthenaPK is updated to be more flexible.
        #        H_mass_fraction = 1.0 - data.ds.parameters["He_mass_fraction"]

        #        cr = lambda_ * (data["gas", "density"] * H_mass_fraction / mh) ** 2

        #        return cr

        #    self.add_field(
        #        ("gas", "cooling_rate"),
        #        sampling_type="cell",
        #        function=_cooling_rate_field,
        #        units=unit_system["energy"]
        #        / unit_system["time"]
        #        / unit_system["length"] ** 3,
        #    )

        #    self.add_field(
        #        ("gas", "cooling_time"),
        #        sampling_type="cell",
        #        function=_cooling_time_field,
        #        units=unit_system["time"],
        #    )
