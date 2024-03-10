import socket
from struct import unpack
import trimesh
import numpy as np 
from Map import Map

class Car:
    def __init__(self, position=[0, 0, 0], direction=[1, 0, 0]) -> None:
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.mesh = trimesh.creation.box(extents=[5, 1, 3], tag="car")
        self.mesh.visual.vertex_colors = [0, 0, 0]

        self.distance = 0
        self.ray = None
    

    def get_mesh(self):
        return self.mesh

    def update(self, new_position, new_direction):
        self.mesh.apply_translation([-self.position[0], -self.position[1], -self.position[2]])

        new_direction = new_direction / np.linalg.norm(new_direction)
        rotation_matrix = trimesh.geometry.align_vectors(self.direction, new_direction)
        self.mesh.apply_transform(rotation_matrix)
        self.direction = np.array(new_direction)

        self.mesh.apply_translation(new_position)
        self.position = np.array(new_position)

    def get_data(self, s: socket.socket, game_map: Map):
        data = dict()
        data['speed'] = unpack(b'@f', s.recv(4))[0] # speed
        data['side_speed'] = unpack(b'@f', s.recv(4))[0] # side speed
        data['distance'] = unpack(b'@f', s.recv(4))[0] # distance
        data['x'] = unpack(b'@f', s.recv(4))[0] # x
        data['y'] = unpack(b'@f', s.recv(4))[0] + 0.5  # y
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


        new_direction = [data['dx'], data['dy'], data['dz']]
        new_position = [data['x'], data['y'], data['z']]
        self.update(new_position, new_direction)
        intersection, distance = self.find_closest_intersection(game_map)
        self.distance = distance
        return data

    def visualize_ray(self, scene: trimesh.Scene):
        if self.ray is not None:
            scene.delete_geometry("ray")
        
        if self.distance == float('inf'):
            self.distance = 320  # Set a default max distance if no collision is detected

        ray_origin = self.position
        ray_direction = self.direction[0], 0, self.direction[2]  # Assuming the ray is on the XZ plane
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_end = ray_origin + ray_direction * self.distance  # Calculate the ray end point

        # Create a line geometry representing the ray
        ray_geometry = trimesh.load_path([ray_origin, ray_end])

        
        # Add the ray geometry to the scene
        self.ray = scene.add_geometry(ray_geometry, node_name="ray")

    
    def find_closest_intersection(self, map: Map):
        ray_origin = self.position
        ray_direction = self.direction[0], 0, self.direction[2]
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
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
        print(min_distance)
        return closest_intersection, min_distance