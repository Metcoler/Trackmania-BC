import socket
import threading
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car

def callback_function(scene: trimesh.Scene):
    global start_time_rendering
    if time() - start_time_rendering < 0.1:
        return
    start_time_rendering = time()
    car.visualize_ray(scene)
    car.update_model_view()
    car.update_camera(scene)


def plot_map():
    scene = trimesh.Scene()
    scene.add_geometry(car.get_mesh())
    scene.add_geometry(game_map.get_mesh())
    scene.show(callback=callback_function)


def print_fps(frame: int):
    global start_time_fps
    if frame % 100 != 0:
        return
    
    dt = time() - start_time_fps
    if dt == 0:
        dt = 0.01
    print(f"FPS: {(100/dt):02f}", end="\r")
    start_time_fps = time()



if __name__ == "__main__":
    vizualize = True
    
    game_map = Map("small_map_test_2")
    car = Car()
    start_time_fps = time()
    start_time_rendering = time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet_socket:
        print("Trying to connect...")
        inet_socket.connect(("127.0.0.1", 9002))
        print("Connected to openplanet")

        if vizualize:
            window_thread = threading.Thread(target=plot_map, daemon=True)
            window_thread.start()

        frame = 0
        while True:
            data = car.get_data(inet_socket, game_map)
            
            print_fps(frame)
            frame += 1
            if vizualize and not window_thread.is_alive():
               break











    