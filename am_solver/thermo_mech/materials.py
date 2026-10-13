import jax.numpy as np


KELVIN_OFFSET = 273.15


class MaterialDatabase:
    """
    Select material by z coordinate and linearly interpolate by temperature.

    Internal solver temperatures use degrees Celsius.
    Temperature tables in YAML use kelvin.
    """

    def __init__(self, config, interface_z):
        self.upper = config["upper_region"]
        self.lower = config["lower_region"]
        self.interface_z = float(interface_z)

    @staticmethod
    def _temperature_c(table):
        return np.asarray(table["temperature_k"], dtype=float) - KELVIN_OFFSET

    @staticmethod
    def _table_values(table, key, factor):
        return np.asarray(table[key], dtype=float) * factor

    def _interpolate(self, temperature_c, z, property_name, value_key="values", factor=1.0):
        upper_table = self.upper[property_name]
        lower_table = self.lower[property_name]

        upper_temperature = self._temperature_c(upper_table)
        lower_temperature = self._temperature_c(lower_table)

        upper_values = self._table_values(upper_table, value_key, factor)

        lower_values = self._table_values(lower_table, value_key, factor)

        upper_result = np.interp(
            temperature_c,
            upper_temperature,
            upper_values,
            left=upper_values[0],
            right=upper_values[-1],
        )

        lower_result = np.interp(
            temperature_c,
            lower_temperature,
            lower_values,
            left=lower_values[0],
            right=lower_values[-1],
        )

        return np.where(z > self.interface_z, upper_result, lower_result)

    def _interpolate_optional(self, temperature_c, z, property_name, value_key="values", factor=1.0, default=0.0):
        def interpolate_region(region):
            table = region.get(property_name)

            if table is None:
                return np.zeros_like(temperature_c) + default

            temperature = self._temperature_c(table)
            values = self._table_values(table, value_key, factor)

            return np.interp(
                temperature_c,
                temperature,
                values,
                left=values[0],
                right=values[-1],
            )

        upper_result = interpolate_region(self.upper)
        lower_result = interpolate_region(self.lower)

        return np.where(z > self.interface_z, upper_result, lower_result)

    def conductivity(self, temperature_c, z):
        return self._interpolate(temperature_c, z, property_name="conductivity")

    def heat_capacity(self, temperature_c, z):
        return self._interpolate(temperature_c, z, property_name="heat_capacity")

    def volumetric_heat_capacity(self, temperature_c, z):
        return self.density(z) * self.heat_capacity(temperature_c, z)

    def elastic_modulus(self, temperature_c, z):
        return self._interpolate(
            temperature_c,
            z,
            property_name="elastic_modulus",
            value_key="values_gpa",
            factor=1.0e9,
        )

    def thermal_expansion(self, temperature_c, z):
        return self._interpolate(
            temperature_c,
            z,
            property_name="thermal_expansion",
            value_key="values_micro_per_k",
            factor=1.0e-6,
        )

    def poisson_ratio(self, temperature_c, z):
        return self._interpolate(temperature_c, z, property_name="poisson_ratio")

    def yield_stress(self, temperature_c, z):
        return self._interpolate(temperature_c, z, property_name="yield_stress", value_key="values_mpa", factor=1.0e6)

    def hardening_modulus(self, temperature_c, z):
        return self._interpolate(
            temperature_c,
            z,
            property_name="hardening_modulus",
            value_key="values_gpa",
            factor=1.0e9,
        )

    def kinematic_hardening_modulus(self, temperature_c, z):
        return self._interpolate_optional(
            temperature_c,
            z,
            property_name="kinematic_hardening_modulus",
            value_key="values_gpa",
            factor=1.0e9,
            default=0.0,
        )

    def maximum_yield_stress(self, temperature_c, z):
        # Prefer an explicit temperature-dependent saturation table when a
        # legacy material provides one; retain the factor-based fallback for
        # existing cases.
        if (
            self.upper.get("maximum_yield_stress") is not None
            and self.lower.get("maximum_yield_stress") is not None
        ):
            return self._interpolate(
                temperature_c,
                z,
                property_name="maximum_yield_stress",
                value_key="values_mpa",
                factor=1.0e6,
            )

        base_yield = self.yield_stress(temperature_c, z)

        upper_factor = float(self.upper.get("maximum_yield_factor", 1.0))

        lower_factor = float(self.lower.get("maximum_yield_factor", 1.0))

        factor = np.where(z > self.interface_z, upper_factor, lower_factor)

        return factor * base_yield

    def density(self, z):
        return np.where(z > self.interface_z, float(self.upper["density_kg_m3"]), float(self.lower["density_kg_m3"]))

    def latent_heat(self, z):
        return np.where(
            z > self.interface_z,
            float(self.upper["latent_heat_j_kg"]),
            float(self.lower["latent_heat_j_kg"]),
        )

    def solidus_temperature(self, z):
        upper_temperature_c = float(self.upper["solidus_temperature_k"]) - KELVIN_OFFSET

        lower_temperature_c = float(self.lower["solidus_temperature_k"]) - KELVIN_OFFSET

        return np.where(z > self.interface_z, upper_temperature_c, lower_temperature_c)

    def liquidus_temperature(self, z):
        upper_temperature_c = float(self.upper["liquidus_temperature_k"]) - KELVIN_OFFSET

        lower_temperature_c = float(self.lower["liquidus_temperature_k"]) - KELVIN_OFFSET

        return np.where(z > self.interface_z, upper_temperature_c, lower_temperature_c)
