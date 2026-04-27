import socket
import sys
import threading
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car
from ObservationEncoder import ObservationEncoder

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
    print(current_fps, end="\r", flush=True)
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
    slices = ObservationEncoder.section_slices(vertical_mode=encoder.vertical_mode)

    lines = [
        summary_line,
        secondary_summary,
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



if __name__ == "__main__":  
    #map_name = "AI Training #2"
    map_name = "loop_test"
    vizualize = False
    vertical_mode = False

    game_map = Map(map_name)
    car = Car(game_map, vertical_mode=vertical_mode)
    encoder = ObservationEncoder(vertical_mode=vertical_mode)

    start_time_fps = time()
    current_fps = 0.0
    
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
        # print_live_debug(frame, data_dictionary, instructions, observation, mirrored_observation)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    
