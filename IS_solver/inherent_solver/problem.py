import jax
import jax.numpy as np

from jax import vmap
from jax_fem.problem import Problem


class InherentStrainPlasticity(Problem):
    def __init__(
        self,
        *args,
        material_config,
        **kwargs,
    ):
        self.elastic_modulus = float(
            material_config[
                "elastic_modulus_pa"
            ]
        )

        self.poisson_ratio = float(
            material_config[
                "poisson_ratio"
            ]
        )

        self.initial_yield_stress = float(
            material_config[
                "initial_yield_stress_pa"
            ]
        )

        self.hardening_modulus = float(
            material_config[
                "hardening_modulus_pa"
            ]
        )

        self.maximum_yield_stress = float(
            material_config[
                "maximum_yield_stress_pa"
            ]
        )

        super().__init__(
            *args,
            **kwargs,
        )

    def custom_init(self):
        self.fe = self.fes[0]

        number_of_cells = len(
            self.fe.cells
        )

        number_of_quadrature_points = (
            self.fe.num_quads
        )

        inherent_strain_increment = np.zeros(
            (
                number_of_cells,
                number_of_quadrature_points,
                3,
            )
        )

        active_mask = np.zeros(
            (
                number_of_cells,
                number_of_quadrature_points,
                1,
            )
        )

        cut_mask = np.zeros_like(
            active_mask
        )

        relaxation_factor = np.zeros_like(
            active_mask
        )

        old_displacement_gradient = np.zeros(
            (
                number_of_cells,
                number_of_quadrature_points,
                self.dim,
                self.dim,
            )
        )

        old_stress = np.zeros_like(
            old_displacement_gradient
        )

        old_plastic_strain = np.zeros(
            (
                number_of_cells,
                number_of_quadrature_points,
            )
        )

        self.internal_vars = [
            inherent_strain_increment,
            active_mask,
            cut_mask,
            relaxation_factor,
            old_displacement_gradient,
            old_stress,
            old_plastic_strain,
        ]

    def get_maps(self):
        elastic_modulus = (
            self.elastic_modulus
        )

        poisson_ratio = (
            self.poisson_ratio
        )

        initial_yield_stress = (
            self.initial_yield_stress
        )

        hardening_modulus = (
            self.hardening_modulus
        )

        maximum_yield_stress = (
            self.maximum_yield_stress
        )

        shear_modulus = (
            elastic_modulus
            / (
                2.0
                * (1.0 + poisson_ratio)
            )
        )

        lame_parameter = (
            elastic_modulus
            * poisson_ratio
            / (
                (1.0 + poisson_ratio)
                * (
                    1.0
                    - 2.0 * poisson_ratio
                )
            )
        )

        def stress_return_map(
            displacement_gradient,
            inherent_strain_increment,
            active_marker,
            cut_marker,
            relaxation_factor,
            old_displacement_gradient,
            old_stress,
            old_plastic_strain,
        ):
            # ==========================================
            # 1. Incremental elastoplasticity for normally active cells
            # ==========================================
            displacement_gradient_increment = (
                displacement_gradient
                - old_displacement_gradient
            )

            geometric_strain_increment = (
                0.5
                * (
                    displacement_gradient_increment
                    + displacement_gradient_increment.T
                )
            )

            inherent_strain_tensor = np.array([
                [
                    inherent_strain_increment[0],
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    inherent_strain_increment[1],
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    inherent_strain_increment[2],
                ],
            ])

            elastic_strain_increment = (
                geometric_strain_increment
                - inherent_strain_tensor
            )

            stress_increment = (
                lame_parameter
                * np.trace(
                    elastic_strain_increment
                )
                * np.eye(self.dim)
                + 2.0
                * shear_modulus
                * elastic_strain_increment
            )

            trial_stress = (
                old_stress
                + stress_increment
            )

            deviatoric_stress = (
                trial_stress
                - (
                    1.0 / self.dim
                    * np.trace(trial_stress)
                    * np.eye(self.dim)
                )
            )

            equivalent_trial_stress = np.sqrt(
                3.0
                / 2.0
                * np.sum(
                    deviatoric_stress
                    * deviatoric_stress
                )
                + 1.0e-14
            )

            uncapped_yield_stress = (
                initial_yield_stress
                + hardening_modulus
                * old_plastic_strain
            )

            is_capped = (
                uncapped_yield_stress
                >= maximum_yield_stress
            )

            current_yield_stress = np.where(
                is_capped,
                maximum_yield_stress,
                uncapped_yield_stress,
            )

            effective_hardening_modulus = np.where(
                is_capped,
                0.0,
                hardening_modulus,
            )

            yield_function = (
                equivalent_trial_stress
                - current_yield_stress
            )

            positive_yield_function = np.where(
                yield_function > 0.0,
                yield_function,
                0.0,
            )

            plastic_strain_increment = (
                positive_yield_function
                / (
                    3.0
                    * shear_modulus
                    + effective_hardening_modulus
                )
            )

            correction = (
                3.0
                * shear_modulus
                * plastic_strain_increment
                * deviatoric_stress
                / equivalent_trial_stress
            )

            elastoplastic_stress = (
                trial_stress
                - correction
            )

            new_plastic_strain = (
                old_plastic_strain
                + plastic_strain_increment
            )

            stress_trace = np.trace(
                elastoplastic_stress
            )

            new_elastic_strain = (
                elastoplastic_stress
                / (2.0 * shear_modulus)
                - (
                    lame_parameter
                    / (
                        2.0
                        * shear_modulus
                        * (
                            3.0
                            * lame_parameter
                            + 2.0
                            * shear_modulus
                        )
                    )
                    * stress_trace
                    * np.eye(self.dim)
                )
            )

            # ==========================================
            # 2. Legacy stress-release interface, retained for now
            # ==========================================
            cut_stress = (
                relaxation_factor[0]
                * old_stress
            )

            active_value = (
                active_marker[0]
            )

            cut_value = (
                cut_marker[0]
            )

            output_stress = (
                active_value
                * elastoplastic_stress
                + cut_value
                * cut_stress
            )

            output_plastic_strain = (
                active_value
                * new_plastic_strain
                + cut_value
                * old_plastic_strain
            )

            output_elastic_strain = (
                active_value
                * new_elastic_strain
            )

            return (
                output_stress,
                output_plastic_strain,
                output_elastic_strain,
            )

        tensor_map = lambda *args: (
            stress_return_map(*args)[0]
        )

        update_map = lambda *args: (
            stress_return_map(*args)
        )

        return (
            tensor_map,
            update_map,
        )

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def set_params(self, params):
        self.internal_vars = params

    def compute_u_grad(self, solution):
        nodal_displacement = np.take(
            solution,
            self.fe.cells,
            axis=0,
        )

        displacement_gradients = (
            nodal_displacement[
                :, None, :, :, None
            ]
            * self.fe.shape_grads[
                :, :, :, None, :
            ]
        )

        return np.sum(
            displacement_gradients,
            axis=2,
        )

    def update_stress_strain(
        self,
        solution,
        params,
    ):
        (
            inherent_strain_increment,
            active_mask,
            cut_mask,
            relaxation_factor,
            old_displacement_gradient,
            old_stress,
            old_plastic_strain,
        ) = params

        displacement_gradients = (
            self.compute_u_grad(
                solution
            )
        )

        _, update_map = self.get_maps()

        vectorized_update = vmap(
            vmap(
                update_map,
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            )
        )

        (
            stress,
            plastic_strain,
            elastic_strain,
        ) = vectorized_update(
            displacement_gradients,
            inherent_strain_increment,
            active_mask,
            cut_mask,
            relaxation_factor,
            old_displacement_gradient,
            old_stress,
            old_plastic_strain,
        )

        return (
            stress,
            plastic_strain,
            elastic_strain,
        )