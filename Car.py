import socket
from struct import unpack
import trimesh
import numpy as np
import threading

from Map import Map


class Car:
    NUM_LASERS = 15
    ANGLE = 180
    def __init__(self, game_map: Map) -> None:
        self.position = np.array(game_map.get_start_position())
        self.direction = np.array(game_map.get_start_direction())
        self.game_map = game_map
        self.speed = 0
        self.side_speed = 0
        self.distance_traveled = 0

        self.last_position = self.position
        self.last_direction = self.direction

        self.mesh = trimesh.creation.box(extents=[5, 1, 3], tag="car")
        rotation_matrix = trimesh.geometry.align_vectors([1, 0, 0], self.direction)
        self.mesh.apply_transform(rotation_matrix)
        self.mesh.apply_translation(self.position)
        self.mesh.visual.vertex_colors = [0, 0, 0]

        self.distances = [0 for _ in range(Car.NUM_LASERS)]
        self.rays_directions = [[1, 0, 0] for _ in range(Car.NUM_LASERS)]
        self.intersections = [[0, 0, 0] for _ in range(Car.NUM_LASERS)]

    
        self.ray_finder = trimesh.ray.ray_triangle.RayMeshIntersector(self.game_map.get_walls_mesh())

        self.data = None
        self.thread = threading.Thread(target=self.data_getter_thread, daemon=True)
        self.thread.start()
    

    def get_mesh(self):
        return self.mesh

    def update_model_view(self):
        self.mesh.apply_translation(-self.last_position)

        rotation_matrix = trimesh.geometry.align_vectors(self.last_direction, self.direction)
        self.mesh.apply_transform(rotation_matrix)
        self.mesh.apply_translation(self.position)
        
        self.last_position = self.position
        self.last_direction = self.direction
    
    def update_camera(self, scene: trimesh.Scene):
        # Calculate the new direction of the camera
        new_direction = np.array([self.direction[0], 0, self.direction[2]])
        new_direction = new_direction / np.linalg.norm(new_direction)

        # Calculate the rotation matrix to align the camera's direction with the positive z-axis
        rotation_matrix = trimesh.geometry.align_vectors([0, 0, -1], new_direction)

        # Calculate the new camera position (move the camera back along its direction and then up by 20 units)
        camera_position = self.position - 32 * self.direction
        camera_position[1] += 20

        # Apply translation to the camera position
        translation_matrix = trimesh.transformations.translation_matrix(camera_position)

        # Calculate rotation to angle the camera down by 20 degrees
        angle_radians = np.radians(20)
        rotation_down_matrix = trimesh.transformations.rotation_matrix(-angle_radians, [1, 0, 0])

        # Combine translation and rotation matrices
        transformation_matrix = translation_matrix @ rotation_matrix @ rotation_down_matrix

        # Set the camera transformation in the scene
        scene.camera_transform = transformation_matrix
    
    def data_getter_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet_socket:
            # Connect to the openplanet plugin
            print("Trying to connect...")
            inet_socket.connect(("127.0.0.1", 9002))
            print("Connected to openplanet")
            
            while True:
                self.data = self.recieve_data_from_openplanet(inet_socket)

    def get_data(self):
        while self.data is None:
            pass
        data = self.data

        self.direction = np.array([data['dx'], 0, data['dz']])
        self.direction = self.direction / np.linalg.norm(self.direction)

        self.position = np.array([data['x'], data['y'], data['z']])
        self.speed = data['speed']
        self.side_speed = data['side_speed']
        self.distance_traveled = data['distance']

        self.generate_laser_directions(Car.ANGLE)
        self.find_closest_intersections()
        return self.distances, data

    def recieve_data_from_openplanet(self, s: socket.socket):
        data = dict()

        ## Wait for newest data
        data['speed'] = unpack(b'@f', s.recv(4))[0] # speed
        data['side_speed'] = unpack(b'@f', s.recv(4))[0] # side speed
        data['distance'] = unpack(b'@f', s.recv(4))[0] # distance
        data['x'] = unpack(b'@f', s.recv(4))[0] # x
        data['y'] = unpack(b'@f', s.recv(4))[0] + 0.2  # y
        data['z'] = unpack(b'@f', s.recv(4))[0] # z
        data['steer'] = unpack(b'@f', s.recv(4))[0] # steer
        data['gas'] = unpack(b'@f', s.recv(4))[0] # gas
        data['brake'] = unpack(b'@f', s.recv(4))[0] # brake
        data['done'] = unpack(b'@f', s.recv(4))[0] # finished
        data['gear'] = unpack(b'@f', s.recv(4))[0] # gear
        data['rpm'] = unpack(b'@f', s.recv(4))[0] # rpm
        data['dx'] = unpack(b'@f', s.recv(4))[0] # dx
        data['dy'] = unpack(b'@f', s.recv(4))[0] # dy
        data['dz'] = unpack(b'@f', s.recv(4))[0] # dz
        data['time'] = unpack(b'@f', s.recv(4))[0] # time

        return data

    def visualize_ray(self, scene: trimesh.Scene):
        
        for i, ray_end in enumerate(self.intersections):
            scene.delete_geometry(f"ray{i}")
        
            ray_origin = self.position
            ray_geometry = trimesh.load_path([ray_origin, ray_end])
            self.ray = scene.add_geometry(ray_geometry, node_name=f"ray{i}")
    
        
    
    def find_closest_intersections(self):
        ray_origin = self.position

        # Perform batch ray intersection test
        hits = self.ray_finder.intersects_location(ray_origins=[ray_origin] * Car.NUM_LASERS,
                                    ray_directions=self.rays_directions,
                                    multiple_hits=False,
                                    parallel=True)

        # Find the closest intersection point for each ray
        num_hits = len(hits[0])
        ray_hits = [False for _ in range(Car.NUM_LASERS)]
        for i in range(num_hits):
            hit_position = hits[0][i]
            ray_index = hits[1][i]
            triengle_hit = hits[2][i]
        
            if len(hit_position) == 0:
                distance = 320
                hit_position = ray_origin + self.rays_directions[ray_index] * distance
            else:
                distance = np.linalg.norm((self.position - hit_position))
            self.intersections[ray_index] = hit_position
            self.distances[ray_index] = distance
            ray_hits[ray_index] = True
        
        for i in range(Car.NUM_LASERS):
            if not ray_hits[i]:
                self.distances[i] = 320
                self.intersections[i] = ray_origin + self.rays_directions[i] * 320
        ##print(self.distances[::-1])
                
    def reset(self):
        self.position = np.array([0, 0, 0])
        self.direction = np.array([1, 0, 0])
        self.speed = 0
        self.side_speed = 0
        self.distance_traveled = 0

        self.last_position = self.position
        self.last_direction = self.direction

        self.distances = [320 for _ in range(Car.NUM_LASERS)]
        self.rays_directions = [[1, 0, 0] for _ in range(Car.NUM_LASERS)]
        self.intersections = [[0, 0, 0] for _ in range(Car.NUM_LASERS)]

    def generate_laser_directions(self, angle_range_degrees):
        """
        Generate unit vectors for laser directions within the specified angular range around the front direction vector.

        Parameters:
        - front_direction (numpy.ndarray): The front direction vector of the car.
        - num_lasers (int): The number of lasers to generate.
        - angle_range_degrees (float): The angular range (in degrees) around the front direction vector.
        """
        # Convert angle range to radians
        angle_range_radians = np.radians(angle_range_degrees)

        # Calculate the angular spacing between lasers
        angular_spacing = angle_range_radians / (Car.NUM_LASERS - 1)

        # Generate unit vectors for laser directions
        laser_directions = []
        for i in range(Car.NUM_LASERS):
            # Calculate the angle offset for this laser
            angle_offset = i * angular_spacing - angle_range_radians / 2

            # Rotate the front direction vector by the angle offset to get the laser direction
            rotation_matrix = trimesh.transformations.rotation_matrix(angle_offset, [0, 1, 0])
            laser_direction = np.dot(rotation_matrix[:3, :3], self.direction)

            # Add the laser direction to the list
            laser_directions.append(laser_direction)
        
        self.rays_directions = laser_directions