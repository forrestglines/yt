from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.physical_constants import kboltz, mh

mag_units = "sqrt(code_mass)/(sqrt(code_length)*code_time)"
pres_units = "code_mass/(code_length*code_time**2)"
rho_units = "code_mass / code_length**3"
vel_units = "code_length / code_time"
mom_units = "code_mass / code_length**2 / code_time"
eng_units = "code_mass / code_length / code_time**2"


def velocity_field(j):
    def _velocity(field, data):
        return data["parthenon", f"MomentumDensity{j}"] / data["parthenon", "Density"]

    return _velocity


class ParthenonFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("Density", (rho_units, ["density"], None)),
        ("MomentumDensity1",(mom_units, ["MomentumDensity1"], None)),
        ("MomentumDensity2",(mom_units, ["MomentumDensity2"], None)),
        ("MomentumDensity3",(mom_units, ["MomentumDensity3"], None)),
        ("TotalEnergyDensity",(eng_units, ["TotalEnergyDensity"], None)),
        ("MagneticField1", (mag_units, [], None)),
        ("MagneticField2", (mag_units, [], None)),
        ("MagneticField3", (mag_units, [], None)),
    )

    def setup_fluid_fields(self):
        from yt.fields.magnetic_field import setup_magnetic_field_aliases

        unit_system = self.ds.unit_system
        # Add velocity fields
        vel_prefix = "velocity"
        for i, comp in enumerate(self.ds.coordinates.axis_order):
            vel_field = ("parthenon", f"Velocity{i+1}")
            mom_field = ("parthenon", f"MomentumDensity{i+1}")
            if vel_field in self.field_list:
                self.add_output_field(
                    vel_field, sampling_type="cell", units="code_length/code_time"
                )
                self.alias(
                    ("gas", f"{vel_prefix}_{comp}"),
                    vel_field,
                    units=unit_system["velocity"],
                )
            elif mom_field in self.field_list:
                self.add_output_field(
                    mom_field,
                    sampling_type="cell",
                    units="code_mass/code_time/code_length**2",
                )
                self.add_field(
                    ("gas", f"{vel_prefix}_{comp}"),
                    sampling_type="cell",
                    function=velocity_field(i + 1),
                    units=unit_system["velocity"],
                )
        # Figure out thermal energy field
        if ("parthenon", "Pressure") in self.field_list:
            self.add_output_field(
                ("parthenon", "Pressure"), sampling_type="cell", units=pres_units
            )
            self.alias(
                ("gas", "pressure"),
                ("parthenon", "Pressure"),
                units=unit_system["pressure"],
            )

            def _specific_thermal_energy(field, data):
                return (
                    data["parthenon", "Pressure"]
                    / (data.ds.gamma - 1.0)
                    / data["parthenon", "Density"]
                )

            self.add_field(
                ("gas", "specific_thermal_energy"),
                sampling_type="cell",
                function=_specific_thermal_energy,
                units=unit_system["specific_energy"],
            )
        elif ("parthenon", "TotalEnergyDensity") in self.field_list:
            self.add_output_field(
                ("parthenon", "TotalEnergyDensity"), sampling_type="cell", units=pres_units
            )

            def _specific_thermal_energy(field, data):
                eint = data["parthenon", "TotalEnergyDensity"] - data["gas", "kinetic_energy_density"]
                if ("parthenon", "B1") in self.field_list:
                    eint -= data["gas", "magnetic_energy_density"]
                return eint / data["parthenon", "Density"]

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

        setup_magnetic_field_aliases(
            self, "parthenon", ["MagneticField%d" % ax for ax in (1, 2, 3)]
        )
