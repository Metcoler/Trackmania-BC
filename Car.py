import select
import socket
import struct
import trimesh
import numpy as np
import threading
import time

from Map import Map
from Map import MAP_BLOCK_SIZE, MAP_GROUND_LEVEL


class Car:
    NUM_LASERS = 15
    ANGLE = 180
    SIGHT_TILES = 10
    PACKET_FLOAT_COUNT = 20
    PACKET_SIZE = PACKET_FLOAT_COUNT * 4

    @staticmethod
    def _normalize_xz(vector) -> np.ndarray | None:
        vec = np.array([vector[0], 0.0, vector[2]], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm

    @classmethod
    def _signed_heading_error(cls, forward_vector, target_vector) -> float:
        forward = cls._normalize_xz(forward_vector)
        target = cls._normalize_xz(target_vector)
        if forward is None or target is None:
            return 0.0

        dot = float(np.clip(np.dot(forward, target), -1.0, 1.0))
        cross_y = float(forward[0] * target[2] - forward[2] * target[0])
        return float(np.arctan2(cross_y, dot) / np.pi)

    def __init__(self, game_map: Map) -> None:
        self.position = np.array(game_map.get_start_position())
        self.direction = np.array(game_map.get_start_direction())
        self.game_map = game_map

        self.path_tile_index = 0
        self.speed = 0
        self.side_speed = 0
        self.distance_traveled = 0
        self.wheel_slips = np.zeros(4, dtype=np.float32)

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
        self.next_tiles = [game_map.start_logical_position for _ in range(Car.SIGHT_TILES)]
        self.next_points = list(map(Map.tile_coordinate_to_point, self.next_tiles))

        self.ray_finder = trimesh.ray.ray_triangle.RayMeshIntersector(self.game_map.get_walls_mesh())

        self.data = None
        self.ready = False
        self.new_data = False
        self.data_lock = threading.Lock()
        self.thread = threading.Thread(target=self.data_getter_thread, daemon=True)
        self.thread.start()

    def get_mesh(self):
        return self.mesh

    def update_model_view(self):
        ## TODO Fix car mesh transformation bug
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
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet_socket:
                    # Connect to the openplanet plugin
                    print("Trying to connect...")
                    inet_socket.connect(("127.0.0.1", 9002))
                    print("Connected to openplanet")
                    self.ready = True

                    recv_buffer = bytearray()
                    while True:
                        readable, _, _ = select.select([inet_socket], [], [], 1.0)
                        if not readable:
                            continue

                        chunk = inet_socket.recv(4096)
                        if not chunk:
                            raise ConnectionError("OpenPlanet socket closed the connection.")
                        recv_buffer.extend(chunk)

                        latest_data = None
                        while len(recv_buffer) >= Car.PACKET_SIZE:
                            packet = bytes(recv_buffer[:Car.PACKET_SIZE])
                            del recv_buffer[:Car.PACKET_SIZE]
                            latest_data = self.decode_data_from_openplanet(packet)

                        if latest_data is None:
                            continue

                        latest_data["recv_time"] = time.perf_counter()
                        with self.data_lock:
                            self.data = latest_data
                            self.new_data = True
            except Exception as e:
                self.ready = False
                print(f"OpenPlanet stream disconnected: {e}")
                time.sleep(1.0)


    def get_data(self):
        while True:
            with self.data_lock:
                if self.data is not None and self.new_data:
                    data = self.data.copy()
                    self.new_data = False
                    break
            time.sleep(0.001)

        if data["time"] < 0:
            self.reset()

        direction = np.array([data['dx'], 0.0, data['dz']], dtype=np.float32)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            self.direction = direction / direction_norm

        self.position = np.array([data['x'], data['y'], data['z']])
        self.speed = data['speed']
        self.side_speed = data['side_speed']
        self.distance_traveled = data['distance']
        self.wheel_slips = np.array(
            [
                data["slip_fl"],
                data["slip_fr"],
                data["slip_rl"],
                data["slip_rr"],
            ],
            dtype=np.float32,
        )
    

        delta_tile = self.update_path_state()
        data['map_progress'] = delta_tile
        data['total_progress'] = (self.path_tile_index / (len(self.game_map.path_tiles)-1)) * 100
        try:
            self.next_tiles = self.game_map.path_tiles[self.path_tile_index:self.path_tile_index + Car.SIGHT_TILES]
            self.next_instructions = self.game_map.path_instructions[self.path_tile_index:self.path_tile_index + Car.SIGHT_TILES]
        except IndexError:
            self.next_tiles = []
            self.next_instructions = []
        
        if len(self.next_tiles) == 0:
            self.next_tiles = [self.game_map.end_logical_position]
        
        if len(self.next_instructions) == 0:
            self.next_instructions = [0]

        if len(self.next_tiles) < Car.SIGHT_TILES:
            self.next_tiles += [self.next_tiles[-1] for _ in range(Car.SIGHT_TILES - len(self.next_tiles))]
        
        if len(self.next_instructions) < Car.SIGHT_TILES:
            self.next_instructions += [self.next_instructions[-1] for _ in range(Car.SIGHT_TILES - len(self.next_instructions))]

        self.next_points = list(map(lambda tile: Map.tile_coordinate_to_point(tile, dy=2), self.next_tiles))
        current_segment_vector = self.next_points[1] - self.next_points[0]
        next_segment_vector = self.next_points[2] - self.next_points[1]
        data["segment_heading_error"] = self._signed_heading_error(
            self.direction,
            current_segment_vector,
        )
        data["next_segment_heading_error"] = self._signed_heading_error(
            self.direction,
            next_segment_vector,
        )
        self.generate_laser_directions(Car.ANGLE)
        self.find_closest_intersections()
        return self.distances, self.next_instructions, data

    

    def update_path_state(self):
        # Check if the car has reached the next path tile

        current_tile = self.position // 32
        current_tile += np.array([0, 9, 0])
        if all(current_tile == self.game_map.path_tiles[self.path_tile_index]):
            return 0

        if self.path_tile_index < len(self.game_map.path_tiles) - 1 and all(current_tile == self.game_map.path_tiles[self.path_tile_index + 1]):
            self.path_tile_index += 1
            return 1
            
        if self.path_tile_index > 0 and all(current_tile == self.game_map.path_tiles[self.path_tile_index - 1]):
            self.path_tile_index -= 1
            return -1
        
        # Probably reset of the car
        if all(current_tile == self.game_map.start_logical_position):
            self.path_tile_index = 0
            return 0
        return 0
        

    def decode_data_from_openplanet(self, packet: bytes):
        if len(packet) != Car.PACKET_SIZE:
            raise ValueError(f"Invalid packet size: {len(packet)} != {Car.PACKET_SIZE}")

        values = struct.unpack("<20f", packet)
        keys = [
            "speed",
            "side_speed",
            "distance",
            "x",
            "y",
            "z",
            "steer",
            "gas",
            "brake",
            "done",
            "gear",
            "rpm",
            "dx",
            "dy",
            "dz",
            "time",
            "slip_fl",
            "slip_fr",
            "slip_rl",
            "slip_rr",
        ]
        data = dict(zip(keys, values))
        data["y"] += 0.2
        return data

    def visualize_rays(self, scene: trimesh.Scene):
        for i, ray_end in enumerate(self.intersections):
            scene.delete_geometry(f"ray{i}")
            ray_origin = self.position
            ray_geometry = trimesh.load_path([ray_origin, ray_end], colors=[[0, 0, 255]])
            scene.add_geometry(ray_geometry, node_name=f"ray{i}")
        
        # for i, next_point in enumerate(self.next_points):
        #    scene.delete_geometry(f"path{i}")
        #    line_geometry = trimesh.load_path([self.position, next_point], colors=[[255, 0, 255]])
        #    
        #    scene.add_geometry(line_geometry, node_name=f"path{i}")

    
        
    
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
        self.wheel_slips = np.zeros(4, dtype=np.float32)

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
