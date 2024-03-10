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

def plot_map():
    game_map.scene.add_geometry(car.get_mesh())
    game_map.scene.show(callback=callback_function)


def print_fps():
    global start_time
    dt = time() - start_time
    if dt == 0:
        dt = 0.01
    print(f"FPS: {1/dt}", end="\r")
    start_time = time()

def update_camera(data):
    new_direction = [data['dx'], 0, data['dz']]
    new_direction = new_direction / np.linalg.norm(new_direction)
    rotation_matrix = trimesh.geometry.align_vectors([0, 0, -1], new_direction)
    transformation_matrix = trimesh.transformations.translation_matrix([data['x'] -32*data['dx'], data['y']+5, data['z'] -32*data['dz']])
    game_map.scene.camera_transform = transformation_matrix @ rotation_matrix


if __name__ == "__main__":
    game_map = Map("Maps/small_map_test_2.txt")
    car = Car()

    start_time = time() 
   
    sleep(0.2) # wait for connection
    print("Waiting to recieve some data...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet_socket:
        print("Trying to connect...")
        inet_socket.connect(("127.0.0.1", 9002))
        print("Connected to openplanet")

        window_thread = threading.Thread(target=plot_map, daemon=False)
        window_thread.start()

        while True:
            data = car.get_data(inet_socket, game_map)
            ##update_camera(data)
            
            
        
            
            ##print_fps()
            
            if not window_thread.is_alive():
                break











    