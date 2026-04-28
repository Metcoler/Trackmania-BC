import socket
import sys
import threading
import shutil
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car
from ObservationEncoder import ObservationEncoder
from SurfaceTypes import surface_material_name

def callback_function(scene: trimesh.Scene):
    # Callback function is called every frame of visualization
    car.visualize_rays(scene)
    car.update_model_view()
    car.update_camera(scene)


def plot_map():
    # Create a scene with the car and the map
    scene = trimesh.Scene()
    scene.add_geometry(car.get_mesh())
    walls_mesh = game_map.get_sensor_walls_mesh() if vertical_mode else game_map.get_walls_mesh()
    scene.add_geometry(walls_mesh)
    scene.add_geometry(game_map.get_road_mesh())
    scene.add_geometry(game_map.get_path_line_mesh())
    scene.add_geometry(game_map.get_path_points_mesh())
    scene.show(callback=callback_function)


def print_fps(frame: int):
    # Print the average FPS every 100 frames
    global start_time_fps, current_fps
    if frame % 100 != 0:
        return
    
    dt = time() - start_time_fps
    if dt == 0:
        dt = 0.01
    current_fps = 100 / dt
    start_time_fps = time()


def format_instruction_window(instructions) -> str:
    values = np.asarray(instructions, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{value:+.0f}" for value in values) + "]"


def format_feature_block(values) -> str:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.array2string(
        vector,
        precision=3,
        suppress_small=False,
        floatmode="fixed",
        max_line_width=240,
    )


def current_surface_summary(data_dictionary) -> tuple[str, float, float, float]:
    wheel_keys = ("fl", "fr", "rl", "rr")
    material_ids = [
        int(round(float(data_dictionary.get(f"ground_material_{wheel}", 80.0))))
        for wheel in wheel_keys
    ]
    material_id = max(set(material_ids), key=material_ids.count)
    material_name = surface_material_name(material_id)
    slips = np.array(
        [float(data_dictionary.get(f"slip_{wheel}", 0.0)) for wheel in wheel_keys],
        dtype=np.float32,
    )
    return material_name, float(np.min(slips)), float(np.mean(slips)), float(np.max(slips))


def build_debug_panel(frame: int, data_dictionary, instructions, observation, mirrored_observation) -> str:
    fps = float(current_fps)
    current_error = float(data_dictionary.get("segment_heading_error", 0.0))
    next_error = float(data_dictionary.get("next_segment_heading_error", 0.0))
    total_progress = float(data_dictionary.get("total_progress", 0.0))
    path_tile_index = int(car.path_tile_index)
    map_progress = float(data_dictionary.get("map_progress", 0.0))
    speed = float(data_dictionary.get("speed", 0.0))
    side_speed = float(data_dictionary.get("side_speed", 0.0))
    game_time = float(data_dictionary.get("time", 0.0))
    vertical_speed = float(data_dictionary.get("vertical_speed", 0.0))
    forward_y = float(data_dictionary.get("forward_y", 0.0))
    vertical_lidar_mode = str(data_dictionary.get("vertical_lidar_mode", "-"))
    support_normal_y = float(data_dictionary.get("support_normal_y", 1.0))
    cross_slope = float(data_dictionary.get("cross_slope", 0.0))
    instructions_text = format_instruction_window(instructions)
    surface_text = format_feature_block(
        data_dictionary.get(
            "next_surface_instructions",
            np.ones(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    height_text = format_feature_block(
        data_dictionary.get(
            "next_height_instructions",
            np.zeros(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    ray_modes = list(data_dictionary.get("ray_debug_modes", []))
    mode_counts = {}
    for value in ray_modes:
        mode_counts[value] = mode_counts.get(value, 0) + 1
    mode_counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(mode_counts.items()))
    summary_line = (
        f"frame={frame:06d}  "
        f"mode={'3D' if encoder.vertical_mode else '2D'}  "
        f"fps={fps:6.2f}  "
        f"time={game_time:6.2f}s  "
        f"progress={total_progress:6.2f}%  "
        f"path_idx={path_tile_index:03d}  "
        f"map_step={map_progress:+.0f}  "
        f"speed={speed:7.2f}  "
        f"side={side_speed:7.2f}  "
        f"seg_err={current_error:+.3f}  "
        f"next_err={next_error:+.3f}  "
        f"instr={instructions_text}  "
        f"surface={surface_text}  "
        f"height={height_text}"
    )
    secondary_summary = (
        f"vertical_speed={vertical_speed:+7.3f}  "
        f"forward_y={forward_y:+6.3f}  "
        f"support_ny={support_normal_y:+6.3f}  "
        f"cross_slope={cross_slope:+6.3f}  "
        f"ray_modes={mode_counts_text if mode_counts_text else '-'}"
    )
    material_name, slip_min, slip_avg, slip_max = current_surface_summary(data_dictionary)
    slices = ObservationEncoder.section_slices(vertical_mode=encoder.vertical_mode)

    lines = [
        summary_line,
        secondary_summary,
        f"current surface: {material_name}  slip min/avg/max={slip_min:.3f}/{slip_avg:.3f}/{slip_max:.3f}",
        "",
        f"obs lasers   : {format_feature_block(observation[slices['lasers']])}",
        f"obs path     : {format_feature_block(observation[slices['path']])}",
        f"obs base     : {format_feature_block(observation[slices['base']])}",
        f"obs slip     : {format_feature_block(observation[slices['slip']])}",
        f"obs surface  : {format_feature_block(observation[slices['surface']])}",
        f"obs height   : {format_feature_block(observation[slices['height']])}",
        f"obs temporal : {format_feature_block(observation[slices['temporal']])}",
        "",
        f"mir lasers   : {format_feature_block(mirrored_observation[slices['lasers']])}",
        f"mir path     : {format_feature_block(mirrored_observation[slices['path']])}",
        f"mir base     : {format_feature_block(mirrored_observation[slices['base']])}",
        f"mir slip     : {format_feature_block(mirrored_observation[slices['slip']])}",
        f"mir surface  : {format_feature_block(mirrored_observation[slices['surface']])}",
        f"mir height   : {format_feature_block(mirrored_observation[slices['height']])}",
        f"mir temporal : {format_feature_block(mirrored_observation[slices['temporal']])}",
    ]
    if "vertical" in slices:
        lines.extend(
            [
                f"obs vertical : {format_feature_block(observation[slices['vertical']])}",
                "",
                f"mir vertical : {format_feature_block(mirrored_observation[slices['vertical']])}",
                f"ray elev raw : {format_feature_block(data_dictionary.get('laser_elevation_rates', np.zeros(Car.NUM_LASERS, dtype=np.float32)))}",
            ]
        )
    return "\n".join(lines)


def print_live_debug(frame: int, data_dictionary, instructions, observation, mirrored_observation) -> None:
    panel = build_debug_panel(
        frame=frame,
        data_dictionary=data_dictionary,
        instructions=instructions,
        observation=observation,
        mirrored_observation=mirrored_observation,
    )
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write(panel)
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_dashboard_lines(frame: int, data_dictionary, instructions, observation) -> list[str]:
    fps = float(current_fps)
    game_time = float(data_dictionary.get("time", 0.0))
    progress = float(data_dictionary.get("total_progress", 0.0))
    speed = float(data_dictionary.get("speed", 0.0))
    side_speed = float(data_dictionary.get("side_speed", 0.0))
    segment_error = float(data_dictionary.get("segment_heading_error", 0.0))
    next_error = float(data_dictionary.get("next_segment_heading_error", 0.0))
    path_tile_index = int(car.path_tile_index)
    instructions_text = format_instruction_window(instructions)
    surface_text = format_feature_block(
        data_dictionary.get(
            "next_surface_instructions",
            np.ones(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    height_text = format_feature_block(
        data_dictionary.get(
            "next_height_instructions",
            np.zeros(Car.SIGHT_TILES, dtype=np.float32),
        )
    )
    wetness = float(data_dictionary.get("wetness", 0.0))
    material_name, slip_min, slip_avg, slip_max = current_surface_summary(data_dictionary)
    support_valid = float(data_dictionary.get("support_valid", 0.0))
    support_normal_y = float(data_dictionary.get("support_normal_y", 1.0))
    forward_y = float(data_dictionary.get("forward_y", 0.0))
    vertical_lidar_mode = str(data_dictionary.get("vertical_lidar_mode", "-"))
    ray_modes = list(data_dictionary.get("ray_debug_modes", []))
    mode_counts = {}
    for value in ray_modes:
        mode_counts[value] = mode_counts.get(value, 0) + 1
    mode_counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(mode_counts.items()))

    lines = [
        f"frame={frame:06d} fps={fps:5.1f} time={game_time:6.2f}s "
        f"prog={progress:6.2f}% idx={path_tile_index:03d} "
        f"spd={speed:7.2f} side={side_speed:7.2f}",
        f"heading_err={segment_error:+.3f}/{next_error:+.3f} "
        f"instr={instructions_text} surface={surface_text} height={height_text} wet={wetness:.2f}",
        f"current_surface={material_name} "
        f"slip_min/avg/max={slip_min:.3f}/{slip_avg:.3f}/{slip_max:.3f}",
        f"vertical lidar={vertical_lidar_mode} support={support_valid:.0f} support_ny={support_normal_y:+.3f} "
        f"forward_y={forward_y:+.3f} ray_modes={mode_counts_text if mode_counts_text else '-'}",
    ]

    if DEBUG_SURFACE_DASHBOARD_ROWS > 4:
        slices = ObservationEncoder.section_slices(vertical_mode=encoder.vertical_mode)
        lines.extend(
            [
                f"obs slip     : {format_feature_block(observation[slices['slip']])}",
                f"obs surface  : {format_feature_block(observation[slices['surface']])}",
                f"obs height   : {format_feature_block(observation[slices['height']])}",
                f"obs temporal : {format_feature_block(observation[slices['temporal']])}",
            ]
        )
    return lines


def print_dashboard_debug(frame: int, data_dictionary, instructions, observation) -> None:
    global previous_dashboard_line_count, last_compact_debug_time

    now = time()
    if now - last_compact_debug_time < DEBUG_PRINT_INTERVAL_SECONDS:
        return
    last_compact_debug_time = now

    lines = build_dashboard_lines(
        frame=frame,
        data_dictionary=data_dictionary,
        instructions=instructions,
        observation=observation,
    )
    terminal_width = max(40, shutil.get_terminal_size((240, 20)).columns)
    clipped_lines = []
    for line in lines:
        if len(line) >= terminal_width:
            line = line[: terminal_width - 4] + "..."
        clipped_lines.append(line.ljust(terminal_width - 1))

    if previous_dashboard_line_count > 0:
        sys.stdout.write(f"\x1b[{previous_dashboard_line_count}F")

    sys.stdout.write("\n".join(clipped_lines))
    sys.stdout.write("\n")
    sys.stdout.flush()
    previous_dashboard_line_count = len(clipped_lines)


if __name__ == "__main__":  
    
    map_name = "surface_test"
    map_name = "height_test"
    #map_name = "AI Training #5"
    vizualize = False
    vertical_mode = True

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    encoder = ObservationEncoder(vertical_mode=vertical_mode)

    start_time_fps = time()
    current_fps = 0.0
    previous_dashboard_line_count = 0
    last_compact_debug_time = 0.0
    DEBUG_PRINT_INTERVAL_SECONDS = 0
    DEBUG_SURFACE_DASHBOARD_ROWS = 4
    
    # Start the visualization thread
    if vizualize:
        window_thread = threading.Thread(target=plot_map, daemon=True)
        window_thread.start()

    # Start the data collection loop
    frame = 0
    while True:
        distances, instructions, data_dictionary = car.get_data()
        observation = encoder.build_observation(distances, instructions, data_dictionary)
        mirrored_observation = ObservationEncoder.mirror_observation(
            observation,
            vertical_mode=encoder.vertical_mode,
        )
        print_fps(frame)
        print_dashboard_debug(frame, data_dictionary, instructions, observation)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    
