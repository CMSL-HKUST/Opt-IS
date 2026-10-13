import jax.numpy as np

from jax_fem.problem import Problem


class Thermal(Problem):
    def __init__(self, *args, materials, heat_source, thermal_config, geometry_config, **kwargs):
        self.materials = materials
        self.heat_source = heat_source

        self.initial_temperature = float(thermal_config["initial_temperature_c"])

        convection_config = thermal_config["convection"]

        self.convection_coefficient = float(convection_config["coefficient_w_m2k"])

        self.convection_ambient_temperature = float(
            convection_config.get("ambient_temperature_c", self.initial_temperature)
        )

        radiation_config = thermal_config["radiation"]

        self.stefan_boltzmann = float(radiation_config["stefan_boltzmann"])

        self.emissivity = float(radiation_config["emissivity"])

        self.radiation_ambient_temperature = float(
            radiation_config.get("ambient_temperature_c", self.initial_temperature)
        )

        self.bottom_z = float(geometry_config["substrate_bottom_z_m"])
        self.bottom_tolerance = float(geometry_config.get("bottom_tolerance_m", 1.0e-4))

        # Optional location-dependent boundary settings used by the thin-wall
        # case. Defaults preserve the original uniform convection behavior.
        self.bottom_convection_coefficient = float(
            convection_config.get("bottom_coefficient_w_m2k", self.convection_coefficient)
        )
        self.bottom_boundary_z = float(
            convection_config.get("bottom_z_m", geometry_config["substrate_bottom_z_m"])
        )
        self.adiabatic_y = convection_config.get("adiabatic_y_m")
        self.adiabatic_tolerance = float(
            convection_config.get("adiabatic_tolerance_m", self.bottom_tolerance)
        )
        self.preserve_case_dirichlet = bool(
            thermal_config.get("preserve_case_dirichlet", False)
        )

        super().__init__(*args, **kwargs)

        self.case_dirichlet_location_fn = (
            self.fe.dirichlet_bc_info[0][0] if self.preserve_case_dirichlet else None
        )

    def custom_init(self):
        self.fe = self.fes[0]

        cell_points = self.fe.points[self.fe.cells]

        shape_values = self.fe.shape_vals

        self.x_quad = np.sum(cell_points[:, None, :, :] * shape_values[None, :, :, None], axis=2)

    def get_tensor_map(self):
        def tensor_map(u_grad, T_old, dt, active_cell_marker, laser_center, switch, T_old_node, x_quad):
            conductivity = self.materials.conductivity(T_old, x_quad[2])

            active_flux = conductivity * u_grad

            return np.where(active_cell_marker, active_flux, np.zeros_like(u_grad))

        return tensor_map

    @staticmethod
    def smooth_liquid_fraction(temperature, solidus_temperature, liquidus_temperature):
        """Return a C1-continuous liquid fraction over the mushy interval.

        The cubic smoothstep retains exactly zero/one outside the phase-change
        range while avoiding the slope jumps of the former piecewise-linear
        law at the solidus and liquidus temperatures.
        """
        interval = np.maximum(liquidus_temperature - solidus_temperature, 1.0e-12)
        normalized_temperature = np.clip(
            (temperature - solidus_temperature) / interval,
            0.0,
            1.0,
        )
        return normalized_temperature**2 * (3.0 - 2.0 * normalized_temperature)

    def get_mass_map(self):
        """Keep only the distributed Goldak source in the standard mass map.

        Transient sensible heat and latent heat are assembled separately in
        ``get_universal_kernel`` with a row-sum lumped capacity. This avoids
        the nodal over/undershoot produced by a consistent capacity matrix for
        a heat source whose radius is comparable to the local mesh spacing.
        """
        def mass_map(T, x, T_old, dt, active_cell_marker, laser_center, switch, T_old_node, x_quad):
            del T_old, dt, T_old_node, x_quad
            heat_source = self.heat_source.volumetric(x, laser_center, switch)
            source_residual = np.zeros_like(T) - heat_source
            return np.where(
                active_cell_marker,
                source_residual,
                np.zeros_like(source_residual),
            )

        return mass_map

    def get_universal_kernel(self):
        """Assemble lumped sensible heat and smooth latent heat per element."""
        def universal_kernel(
            cell_sol_flat,
            x,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            T_old,
            dt,
            active_cell_marker,
            laser_center,
            switch,
            T_old_node,
            x_quad,
        ):
            del x, cell_shape_grads, cell_v_grads_JxW, laser_center, switch

            cell_temperature = self.unflatten_fn_dof(cell_sol_flat)[0]
            shape_values = self.fe.shape_vals
            quadrature_weights = cell_JxW[0]

            # Use the same interpolated current/old temperatures for material
            # properties, while applying their enthalpy increment nodally.
            temperature_quad = np.sum(
                cell_temperature[None, :, :] * shape_values[:, :, None],
                axis=1,
            )
            density = self.materials.density(x_quad[:, 2])
            heat_capacity = self.materials.heat_capacity(
                temperature_quad[:, 0],
                x_quad[:, 2],
            )

            solidus_temperature = self.materials.solidus_temperature(x_quad[:, 2])
            liquidus_temperature = self.materials.liquidus_temperature(x_quad[:, 2])
            liquid_fraction_new = self.smooth_liquid_fraction(
                temperature_quad[:, 0],
                solidus_temperature,
                liquidus_temperature,
            )
            liquid_fraction_old = self.smooth_liquid_fraction(
                T_old[:, 0],
                solidus_temperature,
                liquidus_temperature,
            )
            latent_heat = self.materials.latent_heat(x_quad[:, 2])

            # Row-summing the consistent matrix is equivalent to integrating
            # rho*cp*N_i. The resulting diagonal capacity prevents one heated
            # node from forcing an adjacent node below the ambient temperature.
            lumped_sensible_capacity = np.sum(
                density[:, None]
                * heat_capacity[:, None]
                * shape_values
                * quadrature_weights[:, None],
                axis=0,
            )
            sensible_residual = (
                lumped_sensible_capacity[:, None]
                * (cell_temperature - T_old_node)
                / dt[0]
            )

            # Evaluate the smoothed enthalpy change at quadrature points and
            # project it to the thermal nodes without a slope discontinuity.
            latent_enthalpy_increment = (
                density
                * latent_heat
                * (liquid_fraction_new - liquid_fraction_old)
            )
            latent_residual = np.sum(
                latent_enthalpy_increment[:, None]
                * shape_values
                * quadrature_weights[:, None],
                axis=0,
            )[:, None] / dt[0]

            is_active = np.any(active_cell_marker)
            residual = sensible_residual + latent_residual
            residual = np.where(is_active, residual, np.zeros_like(residual))
            return residual.reshape(-1)

        return universal_kernel

    def get_surface_maps(self):
        def thermal_flux(temperature, point, old_temperature, laser_center, switch, active_face_markers):
            (active_external_face, active_top_face) = active_face_markers

            laser_flux = self.heat_source.surface(point, laser_center, switch)

            laser_flux = np.where(active_top_face, laser_flux, 0.0)

            is_bottom = np.isclose(point[2], self.bottom_boundary_z, atol=self.bottom_tolerance)
            convection_coefficient = np.where(
                is_bottom,
                self.bottom_convection_coefficient,
                self.convection_coefficient,
            )
            convection_flux = convection_coefficient * (
                self.convection_ambient_temperature - old_temperature[0]
            )

            convection_flux = np.where(active_external_face, convection_flux, 0.0)

            ambient_temperature_k = self.radiation_ambient_temperature + 273.15

            surface_temperature_k = old_temperature[0] + 273.15

            radiation_flux = (
                self.stefan_boltzmann
                * self.emissivity
                * (ambient_temperature_k**4 - surface_temperature_k**4)
            )

            radiation_flux = np.where(active_external_face, radiation_flux, 0.0)

            if self.adiabatic_y is not None:
                is_adiabatic = np.isclose(
                    point[1], float(self.adiabatic_y), atol=self.adiabatic_tolerance
                )
                convection_flux = np.where(is_adiabatic, 0.0, convection_flux)
                radiation_flux = np.where(is_adiabatic, 0.0, radiation_flux)

            total_flux = laser_flux + convection_flux + radiation_flux

            return -np.array([total_flux])

        return [thermal_flux]

    def set_params(self, params):
        (
            old_temperature,
            dt,
            laser_center,
            switch,
            quiet_point_mask,
            active_cell_truth_table,
            active_face_truth_table,
        ) = params

        old_temperature_surface = self.fe.convert_from_dof_to_face_quad(old_temperature, self.boundary_inds_list[0])

        old_temperature_body = self.fe.convert_from_dof_to_quad(old_temperature)

        old_temperature_nodes = old_temperature[self.fes[0].cells]

        dt_quad = dt * np.ones((self.fe.num_cells, self.fe.num_quads))

        laser_center_quad_cell = laser_center[None, None, :] * np.ones((self.fe.num_cells, self.fe.num_quads, 1))

        switch_quad_cell = switch * np.ones((self.fe.num_cells, self.fe.num_quads))

        number_of_boundary_faces = len(self.boundary_inds_list[0])

        laser_center_quad_surface = (
            laser_center[None, None, :]
            * np.ones((number_of_boundary_faces, self.fe.num_face_quads, 1))
        )

        switch_quad_surface = switch * np.ones((number_of_boundary_faces, self.fe.num_face_quads))

        active_cell_quad = np.repeat(active_cell_truth_table[:, None], self.fe.num_quads, axis=1)

        active_face_quad = np.repeat(active_face_truth_table[:, None, :], self.fe.num_face_quads, axis=1)

        self.internal_vars = [
            old_temperature_body,
            dt_quad,
            active_cell_quad,
            laser_center_quad_cell,
            switch_quad_cell,
            old_temperature_nodes,
            self.x_quad,
        ]

        self.internal_vars_surfaces = [[
            old_temperature_surface,
            laser_center_quad_surface,
            switch_quad_surface,
            active_face_quad,
        ]]

        self.fe.dirichlet_bc_info[0][0] = get_dirichlet_location_fn(
            quiet_point_mask,
            self.case_dirichlet_location_fn,
        )

        self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)


def get_dirichlet_location_fn(quiet_point_mask, case_location_fn=None):
    # quiet_point_mask always has the total node count, so JIT sees a fixed shape regardless of the number of quiet nodes.
    quiet_point_mask = np.asarray(quiet_point_mask, dtype=bool)

    def dirichlet_location_fn(point, index):
        case_boundary = False if case_location_fn is None else case_location_fn(point)
        return quiet_point_mask[index] | case_boundary

    return dirichlet_location_fn
