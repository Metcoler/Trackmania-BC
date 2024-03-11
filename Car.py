import socket
from struct import unpack
import trimesh
import numpy as np 
from Map import Map

NUM_LASERS = 10

class Car:
    def __init__(self, position=[0, 0, 0], direction=[1, 0, 0]) -> None:
        self.position = np.array(position)
        self.direction = np.array(direction)

        self.last_position = self.position
        self.last_direction = self.direction

        self.mesh = trimesh.creation.box(extents=[5, 1, 3], tag="car")
        self.mesh.visual.vertex_colors = [0, 0, 0]

        self.distances = [0 for _ in range(NUM_LASERS)]
        self.rays = [[1, 0, 0] for _ in range(NUM_LASERS)]
        self.intersections = [[0, 0, 0] for _ in range(NUM_LASERS)]
    

    def get_mesh(self):
        return self.mesh

    def update_model_view(self):
        self.mesh.apply_translation([-self.last_position[0], -self.last_position[1], -self.last_position[2]])

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
    

    def get_data(self, s: socket.socket, game_map: Map):
        data = dict()
        data['speed'] = unpack(b'@f', s.recv(4))[0] # speed
        data['side_speed'] = unpack(b'@f', s.recv(4))[0] # side speed
        data['distance'] = unpack(b'@f', s.recv(4))[0] # distance
        data['x'] = unpack(b'@f', s.recv(4))[0] # x
        data['y'] = unpack(b'@f', s.recv(4))[0] + 0.2  # y
        data['z'] = unpack(b'@f', s.recv(4))[0] # z
        data['steer'] = unpack(b'@f', s.recv(4))[0] # steer
        data['gas'] = unpack(b'@f', s.recv(4))[0] # gas
        data['brake'] = unpack(b'@f', s.recv(4))[0] # brake
        data['packet_number'] = unpack(b'@f', s.recv(4))[0] # finish
        data['gear'] = unpack(b'@f', s.recv(4))[0] # gear
        data['rpm'] = unpack(b'@f', s.recv(4))[0] # rpm
        data['dx'] = unpack(b'@f', s.recv(4))[0] # dx
        data['dy'] = unpack(b'@f', s.recv(4))[0] # dy
        data['dz'] = unpack(b'@f', s.recv(4))[0] # dz


        self.direction = np.array([data['dx'], 0, data['dz']])
        self.direction = self.direction / np.linalg.norm(self.direction)

        self.position = np.array([data['x'], data['y'], data['z']])
    
        
        self.ray_directions = self.generate_laser_directions(180)

        ### THIS SECTION IS VERY SLOW
        ## TODO OPTIMALIZE
        # Perform batch ray casting instead of loop
        self.find_closest_intersections(self.ray_directions, game_map)

        return data

    def visualize_ray(self, scene: trimesh.Scene):
        
        for i, ray_end in enumerate(self.intersections):
            scene.delete_geometry(f"ray{i}")
            
            distance = self.distances[i]
            ray_origin = self.position
            if distance == float('inf'):
                ray_direction = self.ray_directions[i]
                ray_end = ray_origin + ray_direction * distance
                distance = 320

    
            ray_geometry = trimesh.load_path([ray_origin, ray_end])
            self.ray = scene.add_geometry(ray_geometry, node_name=f"ray{i}")
        
    
    def find_closest_intersections(self, ray_directions, map: Map):
        ray_origin = self.position

        # Construct a Trimesh ray object
        ray = trimesh.ray.ray_triangle.RayMeshIntersector(map.mesh)

        # Perform batch ray intersection test
        hits = ray.intersects_location(ray_origins=[ray_origin] * len(ray_directions),
                                    ray_directions=ray_directions,
                                    multiple_hits=False)

        # Find the closest intersection point for each ray
        num_hits = len(hits[0])
        for i in range(num_hits):
            hit_position = hits[0][i]
            ray_index = hits[1][i]
            triengle_hit = hits[2][i]
        
            
            if len(hit_position) == 0:
                distance = float('inf')
            else:
                distance = np.linalg.norm((self.position - hit_position))
            self.intersections[ray_index] = hit_position
            self.distances[ray_index] = distance

    
    def find_closest_intersection(self, ray_direction, map: Map):
        ray_origin = self.position
        
        # Construct a Trimesh ray object
        ray = trimesh.ray.ray_triangle.RayMeshIntersector(map.mesh)

        # Perform ray intersection test
        hits = ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction], multiple_hits=False)

        # Find the closest intersection point
        closest_intersection = None
        min_distance = float('inf')
        for hit_position in hits:
            if len(hit_position) == 0:
                distance = float('inf')
            else:
                distance = np.linalg.norm((self.position - hit_position))

            if distance < min_distance:
                min_distance = distance
                closest_intersection = hit_position # Intersection point

        return closest_intersection, min_distance
    
    def generate_laser_directions(self, angle_range_degrees):
        """
        Generate unit vectors for laser directions within the specified angular range around the front direction vector.

        Parameters:
        - front_direction (numpy.ndarray): The front direction vector of the car.
        - num_lasers (int): The number of lasers to generate.
        - angle_range_degrees (float): The angular range (in degrees) around the front direction vector.

        Returns:
        - List[numpy.ndarray]: A list of unit vectors representing the directions of the lasers.
        """
        num_lasers = NUM_LASERS
        front_direction = self.direction

        # Convert angle range to radians
        angle_range_radians = np.radians(angle_range_degrees)

        # Calculate the angular spacing between lasers
        angular_spacing = angle_range_radians / (num_lasers - 1)

        # Generate unit vectors for laser directions
        laser_directions = []
        for i in range(num_lasers):
            # Calculate the angle offset for this laser
            angle_offset = i * angular_spacing - angle_range_radians / 2

            # Rotate the front direction vector by the angle offset to get the laser direction
            rotation_matrix = trimesh.transformations.rotation_matrix(angle_offset, [0, 1, 0])
            laser_direction = np.dot(rotation_matrix[:3, :3], front_direction)

            # Add the laser direction to the list
            laser_directions.append(laser_direction)

        return laser_directions