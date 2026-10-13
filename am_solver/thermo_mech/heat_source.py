import jax.numpy as np


class GoldakHeatSource:
    def __init__(self, config):
        self.power = float(config["power_w"])
        self.efficiency = float(config["efficiency"])
        self.volumetric_enabled = bool(config.get("volumetric_enabled", True))

        self.front_radius_x = float(config["front_radius_x_m"])

        self.rear_radius_x = float(config["rear_radius_x_m"])

        self.radius_y = float(config["radius_y_m"])

        self.radius_z = float(config["radius_z_m"])

        self.front_fraction = float(config["front_fraction"])

        self.rear_fraction = float(config["rear_fraction"])

        surface_config = config.get("surface_source", {})

        self.surface_enabled = bool(surface_config.get("enabled", False))

        self.surface_radius_x = float(surface_config.get("radius_x_m", self.front_radius_x))

        self.surface_radius_y = float(surface_config.get("radius_y_m", self.radius_y))

    def volumetric(self, point, laser_center, switch):
        if not self.volumetric_enabled:
            return np.asarray(0.0)

        dx = point[0] - laser_center[0]
        dy = point[1] - laser_center[1]
        dz = point[2] - laser_center[2]

        yz_term = dy**2 / self.radius_y**2 + dz**2 / self.radius_z**2

        common_factor = 6.0 * np.sqrt(3.0) / np.pi**1.5

        front_distribution = (
            common_factor
            * self.front_fraction
            / (self.front_radius_x * self.radius_y * self.radius_z)
            * np.exp(-3.0 * (dx**2 / self.front_radius_x**2 + yz_term))
        )

        rear_distribution = (
            common_factor
            * self.rear_fraction
            / (self.rear_radius_x * self.radius_y * self.radius_z)
            * np.exp(-3.0 * (dx**2 / self.rear_radius_x**2 + yz_term))
        )

        distribution = np.where(dx > 0.0, front_distribution, rear_distribution)

        return self.efficiency * self.power * distribution * switch

    def surface(self, point, laser_center, switch):
        if not self.surface_enabled:
            return np.asarray(0.0)

        dx = point[0] - laser_center[0]
        dy = point[1] - laser_center[1]

        normalized_distance = dx**2 / self.surface_radius_x**2 + dy**2 / self.surface_radius_y**2

        distribution = (
            2.0
            / (np.pi * self.surface_radius_x * self.surface_radius_y)
            * np.exp(-2.0 * normalized_distance)
        )

        return self.efficiency * self.power * distribution * switch
