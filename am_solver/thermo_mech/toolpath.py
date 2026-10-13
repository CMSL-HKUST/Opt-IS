from pathlib import Path
import numpy as np


TOOLPATH_COLUMNS = "time, x_start, x_end, y_start, y_end, z, switch"


def _get_first(config, keys, default=None, required=False):
    for key in keys:
        if key in config:
            return config[key]

    if required:
        raise KeyError(f"toolpath.scan 缺少配置项：{' 或 '.join(keys)}")

    return default


def _positive_float(value, name):
    value = float(value)

    if value <= 0.0:
        raise ValueError(f"{name} 必须大于 0")

    return value


def _non_negative_float(value, name):
    value = float(value)

    if value < 0.0:
        raise ValueError(f"{name} 不能小于 0")

    return value


def _validate_toolpath(toolpath):
    if toolpath.ndim == 1:
        toolpath = toolpath[None, :]

    if toolpath.shape[1] != 7:
        raise ValueError(f"打印路径必须包含 7 列：{TOOLPATH_COLUMNS}")

    if toolpath.shape[0] < 2:
        raise ValueError("打印路径至少需要包含起点和一个路径段")

    time_values = toolpath[:, 0]

    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("打印路径 time 列必须严格递增，避免 dt <= 0")

    switches = toolpath[:, 6]

    if np.any((switches != 0.0) & (switches != 1.0)):
        raise ValueError("打印路径 switch 列只能为 0 或 1")

    return toolpath


def _set_initial_toolpath_position(rows):
    """Row zero is a state, not a scan from the center to the first hatch."""
    first_scan = next((row for row in rows[1:] if row[6] == 1), None)
    if first_scan is None:
        raise ValueError("打印路径没有有效的激光扫描段，请检查扫描道数和间距")
    rows[0][1:6] = [
        first_scan[1], first_scan[1], first_scan[3], first_scan[3], first_scan[5]
    ]


def generate_line_toolpath(config):
    number_of_layers = int(config["number_of_layers"])

    if number_of_layers <= 0:
        raise ValueError("toolpath.number_of_layers 必须大于 0")

    layer_height = _positive_float(config["layer_height_m"], "toolpath.layer_height_m")

    cooling_time = _non_negative_float(config["cooling_time_s"], "toolpath.cooling_time_s")

    scan_config = config["scan"]

    scan_length = _positive_float(scan_config["length_m"], "toolpath.scan.length_m")

    scan_velocity = _positive_float(scan_config["velocity_m_s"], "toolpath.scan.velocity_m_s")

    y_position = float(scan_config.get("y_m", 0.0))

    alternate_direction = bool(scan_config.get("alternate_direction", False))

    scan_time = scan_length / scan_velocity

    x_min = -scan_length / 2.0
    x_max = scan_length / 2.0

    rows = []
    current_time = 0.0

    rows.append([current_time, 0.0, 0.0, 0.0, 0.0, 0.0, 0])

    for layer_index in range(1, number_of_layers + 1):
        z_position = layer_index * layer_height

        if alternate_direction and layer_index % 2 == 0:
            x_start = x_max
            x_end = x_min
        else:
            x_start = x_min
            x_end = x_max

        current_time += scan_time

        rows.append([current_time, x_start, x_end, y_position, y_position, z_position, 1])

        current_time += cooling_time

        rows.append([current_time, x_end, x_end, y_position, y_position, z_position, 0])

    _set_initial_toolpath_position(rows)
    return _validate_toolpath(np.asarray(rows, dtype=float))


def generate_box_zigzag_toolpath(config):
    number_of_layers = int(config["number_of_layers"])

    if number_of_layers <= 0:
        raise ValueError("toolpath.number_of_layers 必须大于 0")

    layer_height = _positive_float(config["layer_height_m"], "toolpath.layer_height_m")

    cooling_time = _non_negative_float(config.get("cooling_time_s", 0.0), "toolpath.cooling_time_s")

    include_cooling = bool(config.get("include_cooling", cooling_time > 0.0))

    current_time = float(config.get("start_time_s", 0.0))

    scan_config = config["scan"]

    x_length = _positive_float(
        _get_first(scan_config, ["length_x_m", "x_length_m", "length_m"], required=True),
        "toolpath.scan.length_x_m",
    )

    y_length = _positive_float(
        _get_first(scan_config, ["length_y_m", "y_length_m", "width_m"], default=x_length),
        "toolpath.scan.length_y_m",
    )

    scan_velocity = _positive_float(scan_config["velocity_m_s"], "toolpath.scan.velocity_m_s")

    layer_rotation_degrees = scan_config.get("layer_rotation_degrees")
    first_layer_angle_degrees = float(scan_config.get("first_layer_angle_degrees", 0.0))

    configured_tracks = _get_first(
        scan_config,
        ["tracks_per_layer", "number_of_tracks", "n_tracks"],
        default=None,
    )
    if configured_tracks is not None:
        number_of_tracks = int(configured_tracks)
        if number_of_tracks <= 0:
            raise ValueError("toolpath.scan.tracks_per_layer 必须大于 0")
    if layer_rotation_degrees is not None:
        layer_rotation_degrees = float(layer_rotation_degrees)

        if not np.isfinite(layer_rotation_degrees) or not np.isfinite(first_layer_angle_degrees):
            raise ValueError("toolpath.scan 旋转角度必须为有限数值")

    hatch_spacing = _positive_float(
        _get_first(scan_config, ["hatch_spacing_m", "hatch_space_m"], required=True),
        "toolpath.scan.hatch_spacing_m",
    )

    center_x = float(_get_first(scan_config, ["center_x_m", "center_x"], default=0.0))

    center_y = float(_get_first(scan_config, ["center_y_m", "center_y"], default=0.0))

    rotate_layers = bool(scan_config.get("rotate_layers", True))

    zigzag = bool(scan_config.get("zigzag", True))

    def automatic_track_count(span):
        # Centered tracks with n = floor(span / hatch) leave at least half a
        # hatch between the outermost centerline and either boundary.
        return max(1, int(np.floor((span + 1.0e-12) / hatch_spacing)))

    def centered_track_offsets(count, transverse_span):
        hatch_span = (count - 1) * hatch_spacing
        if hatch_span >= transverse_span:
            raise ValueError(
                "toolpath.scan 扫描道数与间距使横向路径触及或超出区域边界，"
                "无法保留横向边距；请减小 tracks_per_layer 或 hatch_spacing_m"
            )
        return (np.arange(count) - 0.5 * (count - 1)) * hatch_spacing

    if layer_rotation_degrees is None:
        first_layer_direction = str(scan_config.get("first_layer_direction", "x")).lower()

        if first_layer_direction not in ("x", "y"):
            raise ValueError("toolpath.scan.first_layer_direction 只能为 x 或 y")

        # Scan-direction endpoints use the full box extent. Only hatch
        # centerlines in the transverse direction are inset from the boundary.
        number_of_x_scan_tracks = (
            number_of_tracks if configured_tracks is not None
            else automatic_track_count(y_length)
        )
        number_of_y_scan_tracks = (
            number_of_tracks if configured_tracks is not None
            else automatic_track_count(x_length)
        )

    x_min = center_x - x_length / 2.0
    x_max = center_x + x_length / 2.0

    y_min = center_y - y_length / 2.0
    y_max = center_y + y_length / 2.0

    rows = []

    # simulation.py uses row 0 as the initial state; each subsequent row represents a time interval.
    rows.append([current_time, center_x, center_x, center_y, center_y, 0.0, 0])

    def add_segment(duration, x_start, x_end, y_start, y_end, z_position, switch):
        nonlocal current_time

        current_time += float(duration)

        rows.append([current_time, x_start, x_end, y_start, y_end, z_position, switch])

    def clip_rotated_track(unit_x, unit_y, normal_x, normal_y, offset):
        lower_t = -np.inf
        upper_t = np.inf

        for center, normal, direction, lower, upper in (
            (center_x, normal_x, unit_x, x_min, x_max),
            (center_y, normal_y, unit_y, y_min, y_max),
        ):
            coordinate = center + normal * offset

            if abs(direction) < 1.0e-12:
                if coordinate < lower - 1.0e-12 or coordinate > upper + 1.0e-12:
                    return None

                continue

            first_t = (lower - coordinate) / direction
            second_t = (upper - coordinate) / direction
            lower_t = max(lower_t, min(first_t, second_t))
            upper_t = min(upper_t, max(first_t, second_t))

            if upper_t <= lower_t:
                return None

        if not np.isfinite(lower_t) or not np.isfinite(upper_t):
            return None

        return lower_t, upper_t

    for layer_index in range(1, number_of_layers + 1):
        z_position = layer_index * layer_height

        if layer_rotation_degrees is not None:
            angle = np.deg2rad(first_layer_angle_degrees + (layer_index - 1) * layer_rotation_degrees)
            unit_x = float(np.cos(angle))
            unit_y = float(np.sin(angle))
            normal_x = -unit_y
            normal_y = unit_x
            half_span = 0.5 * (abs(normal_x) * x_length + abs(normal_y) * y_length)
            # A standard axis-aligned layer uses the configured track count.
            # Rotated layers are counted from their projected width instead;
            # this allows the rotated rectangle to use a physically sensible
            # number of hatches without forcing every layer to have the same
            # count.  Centering the offsets keeps the outermost path away from
            # the boundary by at least half a hatch when the pitch fits.
            angle_is_axis_aligned = (
                np.isclose(np.sin(angle), 0.0, atol=1.0e-12)
                or np.isclose(np.cos(angle), 0.0, atol=1.0e-12)
            )
            if configured_tracks is not None and angle_is_axis_aligned:
                number_of_layer_tracks = number_of_tracks
            else:
                number_of_layer_tracks = automatic_track_count(2.0 * half_span)
            track_offsets = centered_track_offsets(number_of_layer_tracks, 2.0 * half_span)

            for track_index, offset in enumerate(track_offsets):
                clipped = clip_rotated_track(unit_x, unit_y, normal_x, normal_y, float(offset))

                if clipped is None:
                    continue

                lower_t, upper_t = clipped
                forward = (track_index % 2 == 0) or not zigzag
                start_t, end_t = (lower_t, upper_t) if forward else (upper_t, lower_t)
                x_start = center_x + unit_x * start_t + normal_x * offset
                y_start = center_y + unit_y * start_t + normal_y * offset
                x_end = center_x + unit_x * end_t + normal_x * offset
                y_end = center_y + unit_y * end_t + normal_y * offset
                duration = (upper_t - lower_t) / scan_velocity

                add_segment(duration, x_start, x_end, y_start, y_end, z_position, 1)

        else:
            layer_direction = first_layer_direction

            if rotate_layers and layer_index % 2 == 0:
                layer_direction = "y" if first_layer_direction == "x" else "x"

            if layer_direction == "x":
                y_positions = centered_track_offsets(number_of_x_scan_tracks, y_length) + center_y
                track_duration = x_length / scan_velocity

                for track_index, y_position in enumerate(y_positions):
                    forward = (track_index % 2 == 0) or not zigzag

                    x_start, x_end = (x_min, x_max) if forward else (x_max, x_min)

                    add_segment(track_duration, x_start, x_end, y_position, y_position, z_position, 1)

            else:
                x_positions = centered_track_offsets(number_of_y_scan_tracks, x_length) + center_x
                track_duration = y_length / scan_velocity

                for track_index, x_position in enumerate(x_positions):
                    forward = (track_index % 2 == 0) or not zigzag

                    y_start, y_end = (y_min, y_max) if forward else (y_max, y_min)

                    add_segment(track_duration, x_position, x_position, y_start, y_end, z_position, 1)

        if include_cooling and cooling_time > 0.0:
            # Laser off: retain the last hatch endpoint rather than insert a
            # fictitious jump back to the box center in the recorded path.
            last_x, last_y = rows[-1][2], rows[-1][4]
            add_segment(cooling_time, last_x, last_x, last_y, last_y, z_position, 0)

    _set_initial_toolpath_position(rows)
    return _validate_toolpath(np.asarray(rows, dtype=float))


def generate_toolpath(config):
    pattern = str(config.get("pattern", "line")).lower()

    if pattern in ("line", "single_track"):
        return generate_line_toolpath(config)

    if pattern in ("box_zigzag", "zigzag", "layer_zigzag"):
        return generate_box_zigzag_toolpath(config)

    raise ValueError(f"不支持的 toolpath.pattern：{pattern}")


def load_toolpath_file(config, project_root):
    file_path = Path(config["file"])

    if not file_path.is_absolute():
        file_path = Path(project_root) / file_path

    delimiter = config.get("delimiter", ",")
    skip_rows = int(config.get("skip_rows", 0))

    toolpath = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_rows, dtype=float)

    return _validate_toolpath(toolpath)


def load_toolpath(config, project_root):
    mode = config.get("mode", "generated")

    if mode == "generated":
        return generate_toolpath(config)

    if mode == "file":
        return load_toolpath_file(config, project_root)

    raise ValueError(f"不支持的 toolpath.mode：{mode}")
