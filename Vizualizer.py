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
    scene.add_geometry(game_map.get_walls_mesh())
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


def format_wheel_surface_line(data_dictionary) -> str:
    wheel_keys = ("fl", "fr", "rl", "rr")
    chunks = []
    for wheel in wheel_keys:
        contact = float(data_dictionary.get(f"ground_contact_{wheel}", 0.0))
        material_id = float(data_dictionary.get(f"ground_material_{wheel}", -1.0))
        material_name = surface_material_name(material_id)
        slip = float(data_dictionary.get(f"slip_{wheel}", 0.0))
        icing = float(data_dictionary.get(f"icing_{wheel}", 0.0))
        dirt = float(data_dictionary.get(f"dirt_{wheel}", 0.0))
        chunks.append(
            f"{wheel.upper()}:{material_name}"
            f"#{int(round(material_id)):02d}"
            f" c={contact:.0f}"
            f" slip={slip:.3f}"
            f" ice={icing:.2f}"
            f" dirt={dirt:.2f}"
        )
    wetness = float(data_dictionary.get("wetness", 0.0))
    return " | ".join(chunks) + f" | wet={wetness:.2f}"


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
    support_normal_y = float(data_dictionary.get("support_normal_y", 1.0))
    cross_slope = float(data_dictionary.get("cross_slope", 0.0))
    instructions_text = format_instruction_window(instructions)
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
        f"instr={instructions_text}"
    )
    secondary_summary = (
        f"vertical_speed={vertical_speed:+7.3f}  "
        f"forward_y={forward_y:+6.3f}  "
        f"support_ny={support_normal_y:+6.3f}  "
        f"cross_slope={cross_slope:+6.3f}  "
        f"ray_modes={mode_counts_text if mode_counts_text else '-'}"
    )
    wheel_surface_line = format_wheel_surface_line(data_dictionary)
    slices = ObservationEncoder.section_slices(vertical_mode=encoder.vertical_mode)

    lines = [
        summary_line,
        secondary_summary,
        f"wheel surfaces: {wheel_surface_line}",
        "",
        f"obs lasers   : {format_feature_block(observation[slices['lasers']])}",
        f"obs path     : {format_feature_block(observation[slices['path']])}",
        f"obs base     : {format_feature_block(observation[slices['base']])}",
        f"obs slip     : {format_feature_block(observation[slices['slip']])}",
        f"obs temporal : {format_feature_block(observation[slices['temporal']])}",
        "",
        f"mir lasers   : {format_feature_block(mirrored_observation[slices['lasers']])}",
        f"mir path     : {format_feature_block(mirrored_observation[slices['path']])}",
        f"mir base     : {format_feature_block(mirrored_observation[slices['base']])}",
        f"mir slip     : {format_feature_block(mirrored_observation[slices['slip']])}",
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


def build_compact_debug_line(frame: int, data_dictionary, instructions) -> str:
    fps = float(current_fps)
    game_time = float(data_dictionary.get("time", 0.0))
    progress = float(data_dictionary.get("total_progress", 0.0))
    speed = float(data_dictionary.get("speed", 0.0))
    side_speed = float(data_dictionary.get("side_speed", 0.0))
    segment_error = float(data_dictionary.get("segment_heading_error", 0.0))
    next_error = float(data_dictionary.get("next_segment_heading_error", 0.0))
    path_tile_index = int(car.path_tile_index)
    instructions_text = format_instruction_window(instructions)
    wheel_surface_line = format_wheel_surface_line(data_dictionary)
    return (
        f"frame={frame:06d} fps={fps:5.1f} time={game_time:6.2f}s "
        f"prog={progress:6.2f}% idx={path_tile_index:03d} "
        f"spd={speed:7.2f} side={side_speed:7.2f} "
        f"err={segment_error:+.3f}/{next_error:+.3f} instr={instructions_text} "
        f"| {wheel_surface_line}"
    )


def print_compact_live_debug(frame: int, data_dictionary, instructions) -> None:
    global previous_compact_line_len, last_compact_debug_time

    now = time()
    if now - last_compact_debug_time < DEBUG_PRINT_INTERVAL_SECONDS:
        return
    last_compact_debug_time = now

    line = build_compact_debug_line(
        frame=frame,
        data_dictionary=data_dictionary,
        instructions=instructions,
    )
    terminal_width = max(40, shutil.get_terminal_size((240, 20)).columns)
    if len(line) >= terminal_width:
        line = line[: terminal_width - 4] + "..."
    padding = " " * max(0, previous_compact_line_len - len(line))
    previous_compact_line_len = len(line)
    sys.stdout.write("\r" + line + padding)
    sys.stdout.flush()



if __name__ == "__main__":  
    #map_name = "AI Training #2"
    map_name = "pallete"
    vizualize = True
    vertical_mode = False

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    encoder = ObservationEncoder(vertical_mode=vertical_mode)

    start_time_fps = time()
    current_fps = 0.0
    previous_compact_line_len = 0
    last_compact_debug_time = 0.0
    DEBUG_PRINT_INTERVAL_SECONDS = 0.10
    
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
        print_compact_live_debug(frame, data_dictionary, instructions)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    
