import socket
import threading
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car

def callback_function(scene: trimesh.Scene):
    # Callback function is called every frame of visualization
    global start_time_rendering
    if time() - start_time_rendering < 0.0:
        return
    start_time_rendering = time()
    car.visualize_ray(scene)
    car.update_model_view()
    car.update_camera(scene)


def plot_map():
    # Create a scene with the car and the map
    scene = trimesh.Scene()
    scene.add_geometry(car.get_mesh())
    scene.add_geometry(game_map.get_walls_mesh())
    scene.add_geometry(game_map.get_road_mesh())
    scene.show(callback=callback_function)


def print_fps(frame: int):
    # Print the average FPS every 100 frames
    global start_time_fps
    if frame % 100 != 0:
        return
    
    dt = time() - start_time_fps
    if dt == 0:
        dt = 0.01
    print(f"FPS: {(100/dt):02f}", end="\r")
    start_time_fps = time()



if __name__ == "__main__":  
    map_name = "AI Training #3"
    vizualize = False
    
    game_map = Map(map_name)
    car = Car(game_map)

    # Time variables
    start_time_fps = time()
    start_time_rendering = time()

    
    # Start the visualization thread
    if vizualize:
        window_thread = threading.Thread(target=plot_map, daemon=True)
        window_thread.start()

    # Start the data collection loop
    frame = 0
    while True:
        distances, data_dictionary = car.get_data()
        print_fps(frame)
        frame += 1
        if vizualize and not window_thread.is_alive():
            break











    