import math
from numbers import Real

import jax
import jax.numpy as np
import numpy as onp

from jax_fem.problem import Problem

from thermo_mech.thermal_problem import get_dirichlet_location_fn


class Plasticity(Problem):
    def __init__(self, *args, materials, mechanics_config, phase_config, **kwargs):
        self.materials = materials
        self.reference_temperature = float(mechanics_config["thermal_strain_reference_temperature_c"])
        self.thermal_strain_cutoff = mechanics_config.get("thermal_strain_cutoff")
        # Output-only history. "legacy" preserves the original output and
        # allocates no additional plastic-strain history.
        self.plastic_strain_output = mechanics_config.get("plastic_strain_output", "flow")
        if self.plastic_strain_output not in ("flow", "legacy"):
            raise ValueError("mechanics.plastic_strain_output 必须为 flow 或 legacy")
        self._plastic_strain_diagonal = None

        if self.thermal_strain_cutoff not in (None, "solidus"):
            cutoff = self.thermal_strain_cutoff
            if (
                isinstance(cutoff, bool)
                or not isinstance(cutoff, Real)
                or not math.isfinite(float(cutoff))
                or float(cutoff) <= -273.15
            ):
                raise ValueError(
                    "mechanics.thermal_strain_cutoff 必须为 solidus、null，"
                    "或大于 -273.15 的有限摄氏温度数值"
                )
            self.thermal_strain_cutoff = float(cutoff)

        self.POWDER = int(phase_config.get("powder", 0))
        self.LIQUID = int(phase_config.get("liquid", 1))
        self.SOLID = int(phase_config.get("solid", 2))
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]

        number_of_cells = len(self.fe.cells)
        number_of_quads = self.fe.num_quads
        u_grad_old = np.zeros((number_of_cells, number_of_quads, self.dim, self.dim))
        sigmas_old = np.zeros_like(u_grad_old)
        plastic_strain_old = np.zeros((number_of_cells, number_of_quads))
        back_stress_old = np.zeros_like(u_grad_old)
        if self.plastic_strain_output == "flow":
            self._plastic_strain_diagonal = np.zeros((number_of_cells, number_of_quads, self.dim))
        delta_temperature = np.zeros((number_of_cells, number_of_quads, 1))
        phase = np.ones_like(delta_temperature, dtype=np.int32) * self.POWDER
        active_cell_truth_table = onp.zeros(number_of_cells, dtype=bool)
        active_cell_quad = np.repeat(active_cell_truth_table[:, None], number_of_quads, axis=1)

        cell_points = self.fe.points[self.fe.cells]
        shape_values = self.fe.shape_vals
        x_quad = np.sum(cell_points[:, None, :, :] * shape_values[None, :, :, None], axis=2)
        self.x_quad = x_quad

        initial_temperature = np.zeros((number_of_cells, number_of_quads, 1))
        self.internal_vars = [
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature,
            phase,
            active_cell_quad,
            initial_temperature,
            x_quad,
        ]

    def get_tensor_map(self):
        tensor_map, _, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        def safe_sqrt(value):
            return np.where(value > 0.0, np.sqrt(value), 0.0)

        def safe_divide(numerator, denominator):
            return np.where(denominator == 0.0, 0.0, numerator / denominator)

        def elastic_stress(strain, elastic_modulus, poisson_ratio):
            shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_parameter = elastic_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            return lame_parameter * np.trace(strain) * np.eye(self.dim) + 2.0 * shear_modulus * strain

        def stress_return_maps(
            u_grad,
            u_grad_old,
            sigma_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature,
            phase,
            active_cell_marker,
            temperature,
            x_quad,
            include_plastic_increment=False,
        ):
            temperature = temperature.squeeze(-1)
            delta_temperature = delta_temperature.squeeze(-1)
            phase = phase.squeeze(-1)
            z_coordinate = x_quad[2]

            elastic_modulus_0 = self.materials.elastic_modulus(temperature, z_coordinate)
            initial_yield_stress = self.materials.yield_stress(temperature, z_coordinate)
            hardening_modulus = self.materials.hardening_modulus(temperature, z_coordinate)
            kinematic_hardening_modulus = self.materials.kinematic_hardening_modulus(temperature, z_coordinate)
            maximum_yield_stress = self.materials.maximum_yield_stress(temperature, z_coordinate)
            thermal_expansion_0 = self.materials.thermal_expansion(temperature, z_coordinate)
            poisson_ratio = self.materials.poisson_ratio(temperature, z_coordinate)

            if self.thermal_strain_cutoff is not None:
                # "solidus" retains the material-dependent cutoff. A numeric
                # cutoff is in degC and changes only the thermal-strain cap,
                # not phase transitions, latent heat, or temperature evolution.
                # Clip both endpoints so heating/cooling across the cap are
                # treated symmetrically; keep all mechanical history intact.
                thermal_strain_cutoff = (
                    self.materials.solidus_temperature(z_coordinate)
                    if self.thermal_strain_cutoff == "solidus"
                    else self.thermal_strain_cutoff
                )
                previous_temperature = temperature - delta_temperature
                effective_previous_temperature = np.minimum(
                    previous_temperature,
                    thermal_strain_cutoff,
                )
                effective_temperature = np.minimum(
                    temperature,
                    thermal_strain_cutoff,
                )
                effective_delta_temperature = (
                    effective_temperature - effective_previous_temperature
                )
                thermal_expansion = thermal_expansion_0
            else:
                effective_temperature = temperature
                effective_delta_temperature = delta_temperature
                thermal_expansion = np.where(
                    phase != self.LIQUID,
                    thermal_expansion_0,
                    0.0,
                )

            elastic_modulus = elastic_modulus_0
            shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))

            delta_u_grad = u_grad - u_grad_old
            strain_increment = 0.5 * (delta_u_grad + delta_u_grad.T)
            thermal_strain_increment = (
                thermal_expansion
                * effective_delta_temperature
                * np.eye(self.dim)
            )
            trial_stress = elastic_stress(strain_increment - thermal_strain_increment, elastic_modulus, poisson_ratio) + sigma_old
            trial_deviatoric_stress = trial_stress - (1.0 / self.dim * np.trace(trial_stress) * np.eye(self.dim))
            shifted_trial_stress = trial_deviatoric_stress - back_stress_old
            equivalent_trial_stress = safe_sqrt(3.0 / 2.0 * np.sum(shifted_trial_stress * shifted_trial_stress))

            uncapped_yield_stress = initial_yield_stress + hardening_modulus * plastic_strain_old
            is_capped = uncapped_yield_stress >= maximum_yield_stress
            current_yield_stress = np.where(is_capped, maximum_yield_stress, uncapped_yield_stress)
            effective_hardening_modulus = np.where(is_capped, 0.0, hardening_modulus)
            yield_function = equivalent_trial_stress - current_yield_stress
            positive_yield_function = np.where(yield_function > 0.0, yield_function, 0.0)
            plastic_strain_increment = positive_yield_function / (
                3.0 * shear_modulus + effective_hardening_modulus + kinematic_hardening_modulus
            )

            correction = safe_divide(
                3.0 * shear_modulus * plastic_strain_increment * shifted_trial_stress,
                equivalent_trial_stress,
            )
            stress = trial_stress - correction
            active_stress = np.where(active_cell_marker, stress, np.zeros_like(stress))
            plastic_strain_new = np.where(active_cell_marker, plastic_strain_old + plastic_strain_increment, 0.0)

            back_stress_increment = safe_divide(
                kinematic_hardening_modulus * plastic_strain_increment * shifted_trial_stress,
                equivalent_trial_stress,
            )
            back_stress_new = np.where(active_cell_marker, back_stress_old + back_stress_increment, 0.0)

            updated_deviatoric_stress = stress - (1.0 / self.dim * np.trace(stress) * np.eye(self.dim))
            equivalent_stress_updated = safe_sqrt(
                3.0 / 2.0 * np.sum(updated_deviatoric_stress * updated_deviatoric_stress)
            )
            equivalent_stress_updated = np.where(
                yield_function > 0.0,
                equivalent_stress_updated,
                safe_sqrt(3.0 / 2.0 * np.sum(trial_deviatoric_stress * trial_deviatoric_stress)),
            )
            equivalent_stress_updated = np.where(active_cell_marker, equivalent_stress_updated, 0.0)

            total_strain = 0.5 * (u_grad + u_grad.T)
            total_strain = np.where(active_cell_marker, total_strain, np.zeros_like(total_strain))
            elastic_strain = (
                (1.0 + poisson_ratio) / elastic_modulus * active_stress
                - (poisson_ratio / elastic_modulus * np.trace(active_stress) * np.eye(self.dim))
            )
            elastic_strain = np.where(active_cell_marker, elastic_strain, np.zeros_like(elastic_strain))

            thermal_strain = (
                thermal_expansion
                * (effective_temperature - self.reference_temperature)
                * np.eye(self.dim)
            )
            plastic_strain_tensor = np.where(
                active_cell_marker,
                total_strain - elastic_strain - thermal_strain,
                np.zeros_like(total_strain),
            )
            mechanical_strain = np.where(active_cell_marker, total_strain - thermal_strain, np.zeros_like(total_strain))

            output_information = (
                positive_yield_function,
                active_stress[0, 0],
                active_stress[1, 1],
                active_stress[2, 2],
                equivalent_stress_updated,
                total_strain[0, 0],
                total_strain[1, 1],
                total_strain[2, 2],
                elastic_strain[0, 0],
                elastic_strain[1, 1],
                elastic_strain[2, 2],
                plastic_strain_tensor[0, 0],
                plastic_strain_tensor[1, 1],
                plastic_strain_tensor[2, 2],
                mechanical_strain[0, 0],
                mechanical_strain[1, 1],
                mechanical_strain[2, 2],
            )

            if include_plastic_increment:
                # Reuse this step's stress correction: correction = 2G * d_eps_p.
                # This branch runs only for post-convergence output, never in
                # tensor_map/update_map or their stress/tangent calculations.
                plastic_diagonal_increment = safe_divide(np.diag(correction), 2.0 * shear_modulus)
                plastic_diagonal_increment = np.where(
                    active_cell_marker, plastic_diagonal_increment, np.zeros_like(plastic_diagonal_increment)
                )
                output_information = output_information + (plastic_diagonal_increment,)

            return active_stress, plastic_strain_new, back_stress_new, output_information

        tensor_map = lambda *args: stress_return_maps(*args)[0]
        update_map = lambda *args: (stress_return_maps(*args)[0], stress_return_maps(*args)[1], stress_return_maps(*args)[2])
        information_map = lambda *args: stress_return_maps(
            *args, include_plastic_increment=self.plastic_strain_output == "flow"
        )[3]

        return tensor_map, update_map, information_map

    def vmap_stress_strain_fns(self):
        _, update_map, information_map = self.get_maps()
        return jax.vmap(jax.vmap(update_map)), jax.vmap(jax.vmap(information_map))

    def compute_u_grad(self, solution):
        nodal_displacements = np.take(solution, self.fe.cells, axis=0)
        displacement_gradients = nodal_displacements[:, None, :, :, None] * self.fe.shape_grads[:, :, :, None, :]
        return np.sum(displacement_gradients, axis=2)

    def update_stress_strain(self, solution, params):
        u_gradients = self.compute_u_grad(solution)
        vmap_update, vmap_information = self.vmap_stress_strain_fns()
        (
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature,
            phase,
            quiet_point_mask,
            active_cell_truth_table,
            temperature,
            x_quad,
        ) = params

        sigmas_updated, plastic_strain_updated, back_stress_updated = vmap_update(
            u_gradients,
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature,
            phase,
            active_cell_truth_table,
            temperature,
            x_quad,
        )
        plastic_information = vmap_information(
            u_gradients,
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature,
            phase,
            active_cell_truth_table,
            temperature,
            x_quad,
        )

        if self.plastic_strain_output == "flow":
            # Commit once per converged mechanical step. This history is NOT
            # passed to set_params/internal_vars and cannot affect equilibrium.
            increment = plastic_information[17]
            old_diagonal = self._plastic_strain_diagonal
            if old_diagonal is None:
                old_diagonal = np.zeros_like(increment)
            self._plastic_strain_diagonal = np.where(
                active_cell_truth_table[..., None], old_diagonal + increment, np.zeros_like(increment)
            )
            # Keep all 17 existing output quantities unchanged; append only
            # the three accumulated tensorial normal strains (dimensionless).
            plastic_information = plastic_information[:17] + tuple(
                self._plastic_strain_diagonal[..., axis] for axis in range(self.dim)
            )

        updated_params = [
            u_gradients,
            sigmas_updated,
            plastic_strain_updated,
            back_stress_updated,
            delta_temperature,
            phase,
            quiet_point_mask,
            active_cell_truth_table,
            temperature,
            x_quad,
        ]
        return updated_params, plastic_information

    def update_dT_and_phase(self, delta_temperature, temperature, params):
        (
            u_grad_old,
            sigmas,
            plastic_strain,
            back_stress,
            _,
            phase,
            quiet_point_mask,
            active_cell_truth_table,
            _,
            x_quad,
        ) = params

        delta_temperature_quad = self.fe.convert_from_dof_to_quad(delta_temperature)
        temperature_quad = self.fe.convert_from_dof_to_quad(temperature)
        active_cell_marker = active_cell_truth_table[:, :, None]
        z_quad = x_quad[..., 2]
        solidus_temperature = self.materials.solidus_temperature(z_quad)[..., None]

        powder_to_liquid = (phase == self.POWDER) & (temperature_quad > solidus_temperature)
        liquid_to_solid = (phase == self.LIQUID) & (temperature_quad < solidus_temperature)
        phase = np.where(powder_to_liquid, self.LIQUID, phase)
        phase = np.where(liquid_to_solid, self.SOLID, phase)
        phase = np.where(active_cell_marker, phase, np.zeros_like(phase))

        return [
            u_grad_old,
            sigmas,
            plastic_strain,
            back_stress,
            delta_temperature_quad,
            phase,
            quiet_point_mask,
            active_cell_truth_table,
            temperature_quad,
            x_quad,
        ]

    def set_params(self, params):
        (
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature_quad,
            phase,
            quiet_point_mask,
            active_cell_truth_table,
            temperature_quad,
            x_quad,
        ) = params

        self.internal_vars = [
            u_grad_old,
            sigmas_old,
            plastic_strain_old,
            back_stress_old,
            delta_temperature_quad,
            phase,
            active_cell_truth_table,
            temperature_quad,
            x_quad,
        ]

        # Pass a fixed-length mask rather than a variable-length quiet-index array to avoid repeated JIT compilation during mechanical solves.
        quiet_location_function = get_dirichlet_location_fn(quiet_point_mask)
        self.fe.dirichlet_bc_info[0][0] = quiet_location_function
        self.fe.dirichlet_bc_info[0][1] = quiet_location_function
        self.fe.dirichlet_bc_info[0][2] = quiet_location_function
        self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)
