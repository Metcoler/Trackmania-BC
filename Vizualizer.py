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
    instructions_text = format_instruction_window(instructions)
    summary_line = (
        f"frame={frame:06d}  "
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
    laser_end = Car.NUM_LASERS
    instruction_end = laser_end + Car.SIGHT_TILES
    base_end = instruction_end + len(ObservationEncoder.BASE_FEATURE_NAMES)
    slip_end = base_end + len(ObservationEncoder.WHEEL_SLIP_FEATURE_NAMES)
    temporal_end = slip_end + len(ObservationEncoder.TEMPORAL_FEATURE_NAMES)

    lines = [
        summary_line,
        "",
        f"obs lasers   : {format_feature_block(observation[:laser_end])}",
        f"obs path     : {format_feature_block(observation[laser_end:instruction_end])}",
        f"obs base     : {format_feature_block(observation[instruction_end:base_end])}",
        f"obs slip     : {format_feature_block(observation[base_end:slip_end])}",
        f"obs temporal : {format_feature_block(observation[slip_end:temporal_end])}",
        "",
        f"mir lasers   : {format_feature_block(mirrored_observation[:laser_end])}",
        f"mir path     : {format_feature_block(mirrored_observation[laser_end:instruction_end])}",
        f"mir base     : {format_feature_block(mirrored_observation[instruction_end:base_end])}",
        f"mir slip     : {format_feature_block(mirrored_observation[base_end:slip_end])}",
        f"mir temporal : {format_feature_block(mirrored_observation[slip_end:temporal_end])}",
    ]
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
    map_name = "AI Training #3"
    vizualize = True

    game_map = Map(map_name)
    car = Car(game_map)
    encoder = ObservationEncoder()

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
        mirrored_observation = ObservationEncoder.mirror_observation(observation)
        print_fps(frame)
        print_live_debug(frame, data_dictionary, instructions, observation, mirrored_observation)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    
