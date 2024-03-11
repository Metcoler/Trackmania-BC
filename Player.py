import socket
from struct import unpack
import threading
from time import sleep, time
import trimesh
import numpy as np
from Map import Map
from Car import Car

def callback_function(scene: trimesh.Scene):
    car.visualize_ray(scene)
    car.update_model_view()
    car.update_camera(scene)


def plot_map():
    game_map.scene.add_geometry(car.get_mesh())
    game_map.scene.show(callback=callback_function)


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
    game_map = Map("Maps/AI Training.txt")
    car = Car()
    start_time_fps = time() 
   
    sleep(0.2) # wait for connection
    print("Waiting to recieve some data...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet_socket:
        print("Trying to connect...")
        inet_socket.connect(("127.0.0.1", 9002))
        print("Connected to openplanet")

        window_thread = threading.Thread(target=plot_map, daemon=False)
        window_thread.start()
        frame = 0
        while True:
            data = car.get_data(inet_socket, game_map)
            
            print_fps(frame)
            frame += 1
            if not window_thread.is_alive():
               break











    