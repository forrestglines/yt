from yt._typing import KnownFieldsT
from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.physical_constants import kboltz, mh, amu
import numpy as np


mag_units = "code_magnetic"
pres_units = "code_mass/(code_length*code_time**2)"
rho_units = "code_mass / code_length**3"
vel_units = "code_length / code_time"
mom_units = "code_mass / code_length**2 / code_time"
eng_units = "code_mass / code_length / code_time**2"


def velocity_field(j):
    def _velocity(field, data):
        return data["parthenon", f"MomentumDensity{j}"] / data["parthenon", "Density"]

    return _velocity

def _cooling_time_field(field, data):

    cooling_time = data['Density'] * data['specific_thermal_energy'] / data['cooling_rate']

    #Set cooling time where Cooling_Rate==0 to infinity
    inf_ct_mask = data['cooling_rate'] == 0
    cooling_time[ inf_ct_mask ] = data.ds.quan(np.inf,"s")

    return cooling_time

class ParthenonFieldInfo(FieldInfoContainer):
    known_other_fields: KnownFieldsT = (
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
                #TODO This only accounts for ideal gases with adiabatic indices
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
                if ("parthenon", "MagneticField1") in self.field_list:
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

        if "cooling_table_filename" in self.ds.specified_parameters:
            #A cooling table is provided - compute "Cooling_Rate" and "Cooling_Time"

            cooling_table = np.loadtxt(self.ds.specified_parameters["cooling_table_filename"])
            log_temps   = cooling_table[:,self.ds.specified_parameters["cooling_table_log_temp_col"]]
            log_lambdas = cooling_table[:,self.ds.specified_parameters["cooling_table_log_lambda_col"]]

            lambdas_units = self.ds.quan(self.ds.specified_parameters["cooling_table_lambda_units_cgs"],"erg*cm**3/s")

            def _cooling_rate_field(field, data):
                nonlocal log_temps, log_lambdas, lambdas_units
                log_temp = np.log10(data["gas","temperature"].in_units("K").v)
                log_lambda = np.interp(log_temp,log_temps,log_lambdas)

                #Zero cooling below the table
                log_lambda[ log_temp < log_temps[0] ] = 0

                #Interpolate free-free cooling (~T^1/2) above the table
                log_lambda[log_temp > log_temps[-1]] = 0.5*log_temp[log_temp > log_temps[-1]] - 0.5*log_temps[-1] + log_lambdas[-1];

                lambda_ = 10**(log_lambda)*lambdas_units

                H_mass_fraction = data.ds.parameters["H_mass_fraction"]

                cr = lambda_*(data["gas","density"]*H_mass_fraction/amu)**2

                return cr

            self.add_field(
                ("gas", "cooling_rate"),
                sampling_type="cell",
                function=_cooling_rate_field,
                units=unit_system["energy"]/unit_system["time"]/unit_system["length"]**3,
            )

            self.add_field(
                ("gas", "cooling_time"),
                sampling_type="cell",
                function=_cooling_time_field,
                units=unit_system["time"],
            )


