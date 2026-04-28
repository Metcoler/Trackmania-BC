import select
import socket
import struct
import trimesh
import numpy as np
import threading
import time
from trimesh.triangles import points_to_barycentric

from Map import Map
from Map import MAP_BLOCK_SIZE, MAP_GROUND_LEVEL


class Car:
    NUM_LASERS = 15
    ANGLE = 180
    SIGHT_TILES = 5
    PACKET_FLOAT_COUNT = 37
    PACKET_SIZE = PACKET_FLOAT_COUNT * 4
    LASER_MAX_DISTANCE = 160.0
    SURFACE_STEP_SIZE = 5.0
    SURFACE_PROBE_HEIGHT = 1.0
    SURFACE_RAY_LIFT = 1.0
    SURFACE_TRAVERSAL_EPS = 1e-4
    SURFACE_WALL_CAST_EPS = 1e-4

    @staticmethod
    def _normalize(vector) -> np.ndarray | None:
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm

    @staticmethod
    def _normalize_xz(vector) -> np.ndarray | None:
        vec = np.array([vector[0], 0.0, vector[2]], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm

    @classmethod
    def _project_onto_plane(cls, vector, plane_normal) -> np.ndarray | None:
        vec = np.asarray(vector, dtype=np.float32)
        normal = cls._normalize(plane_normal)
        if normal is None:
            return cls._normalize(vec)
        projected = vec - float(np.dot(vec, normal)) * normal
        return cls._normalize(projected)

    @classmethod
    def _signed_heading_error(cls, forward_vector, target_vector) -> float:
        forward = cls._normalize_xz(forward_vector)
        target = cls._normalize_xz(target_vector)
        if forward is None or target is None:
            return 0.0

        dot = float(np.clip(np.dot(forward, target), -1.0, 1.0))
        cross_y = float(forward[0] * target[2] - forward[2] * target[0])
        return float(np.arctan2(cross_y, dot) / np.pi)

    def __init__(
        self,
        game_map: Map,
        vertical_mode: bool = False,
        surface_step_size: float = SURFACE_STEP_SIZE,
        surface_probe_height: float = SURFACE_PROBE_HEIGHT,
        surface_ray_lift: float = SURFACE_RAY_LIFT,
    ) -> None:
        self.position = np.array(game_map.get_start_position())
        self.direction = np.array(game_map.get_start_direction())
        initial_direction_world = self._normalize(self.direction)
        if initial_direction_world is None:
            initial_direction_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.direction_world = initial_direction_world
        self.game_map = game_map
        self.road_mesh = self.game_map.get_road_mesh()
        self.road_traversal_mesh = self.game_map.get_road_traversal_mesh()
        self.road_face_neighbors = self.game_map.get_road_face_neighbors()
        self.walls_mesh = self.game_map.get_walls_mesh()
        self.vertical_mode = bool(vertical_mode)
        self.surface_step_size = float(surface_step_size)
        self.surface_probe_height = float(surface_probe_height)
        self.surface_ray_lift = float(surface_ray_lift)

        self.path_tile_index = 0
        self.speed = 0
        self.side_speed = 0
        self.distance_traveled = 0
        self.wheel_slips = np.zeros(4, dtype=np.float32)
        self.surface_support_point = np.array(self.position, dtype=np.float32)
        self.surface_support_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.surface_support_face_index = -1
        self.surface_forward = np.array(self.direction_world, dtype=np.float32)
        self.support_valid = False
        self.ray_paths = [[np.array(self.position, dtype=np.float32)] for _ in range(Car.NUM_LASERS)]
        self.ray_debug_modes = ["flat_open" for _ in range(Car.NUM_LASERS)]
        self.laser_elevation_rates = np.zeros(Car.NUM_LASERS, dtype=np.float32)
        self.laser_elevation_deltas = np.zeros(Car.NUM_LASERS, dtype=np.float32)
        self.laser_surface_lengths = np.zeros(Car.NUM_LASERS, dtype=np.float32)

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

        self.wall_ray_finder = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.walls_mesh
        )
        self.road_ray_finder = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.road_traversal_mesh
        )
        self.ray_finder = self.wall_ray_finder

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

    @staticmethod
    def _collect_nearest_hits(hit_positions, hit_rays, hit_faces, ray_origins):
        ray_origins = np.asarray(ray_origins, dtype=np.float32)
        nearest_hits = [None] * len(ray_origins)
        for hit_position, ray_index, face_index in zip(hit_positions, hit_rays, hit_faces):
            origin = ray_origins[int(ray_index)]
            distance = float(np.linalg.norm(np.asarray(hit_position, dtype=np.float32) - origin))
            current = nearest_hits[int(ray_index)]
            if current is None or distance < current["distance"]:
                nearest_hits[int(ray_index)] = dict(
                    position=np.asarray(hit_position, dtype=np.float32),
                    face_index=int(face_index),
                    distance=distance,
                )
        return nearest_hits

    def _batch_ray_hits(self, intersector, ray_origins, ray_directions):
        ray_origins = np.asarray(ray_origins, dtype=np.float32)
        ray_directions = np.asarray(ray_directions, dtype=np.float32)
        if ray_origins.size == 0:
            return []
        hits = intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
            parallel=True,
        )
        return self._collect_nearest_hits(
            hit_positions=hits[0],
            hit_rays=hits[1],
            hit_faces=hits[2],
            ray_origins=ray_origins,
        )

    def _find_surface_support(
        self,
        target_points: np.ndarray,
        reference_normals: np.ndarray,
    ) -> list[dict | None]:
        target_points = np.asarray(target_points, dtype=np.float32)
        reference_normals = np.asarray(reference_normals, dtype=np.float32)
        if target_points.ndim == 1:
            target_points = target_points.reshape(1, 3)
        if reference_normals.ndim == 1:
            reference_normals = np.repeat(reference_normals.reshape(1, 3), target_points.shape[0], axis=0)

        valid_normals = []
        for normal in reference_normals:
            normalized = self._normalize(normal)
            if normalized is None:
                normalized = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            valid_normals.append(normalized)
        valid_normals = np.asarray(valid_normals, dtype=np.float32)

        normal_origins = target_points + valid_normals * self.surface_probe_height
        normal_directions = -valid_normals
        world_origins = target_points + np.array([0.0, self.surface_probe_height, 0.0], dtype=np.float32)
        world_directions = np.repeat(
            np.array([[0.0, -1.0, 0.0]], dtype=np.float32),
            target_points.shape[0],
            axis=0,
        )

        normal_hits = self._batch_ray_hits(
            self.road_ray_finder,
            normal_origins,
            normal_directions,
        )
        world_hits = self._batch_ray_hits(
            self.road_ray_finder,
            world_origins,
            world_directions,
        )

        supports: list[dict | None] = []
        max_probe_distance = (2.0 * self.surface_probe_height) + 1.0
        for index in range(target_points.shape[0]):
            support = None
            hit = normal_hits[index] if index < len(normal_hits) else None
            if hit is not None and hit["distance"] <= max_probe_distance:
                face_normal = self._normalize(
                    self.road_traversal_mesh.face_normals[hit["face_index"]]
                )
                if face_normal is not None:
                    support = dict(
                        point=hit["position"],
                        normal=face_normal,
                        face_index=int(hit["face_index"]),
                        mode="normal_probe",
                    )
            if support is None:
                hit = world_hits[index] if index < len(world_hits) else None
                if hit is not None and hit["distance"] <= max_probe_distance:
                    face_normal = self._normalize(
                        self.road_traversal_mesh.face_normals[hit["face_index"]]
                    )
                    if face_normal is not None:
                        support = dict(
                            point=hit["position"],
                            normal=face_normal,
                            face_index=int(hit["face_index"]),
                            mode="world_probe",
                        )
            supports.append(support)
        return supports

    def _update_surface_state(self):
        fallback_normal = self.surface_support_normal if self.support_valid else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        support = self._find_surface_support(
            target_points=np.asarray([self.position], dtype=np.float32),
            reference_normals=np.asarray([fallback_normal], dtype=np.float32),
        )[0]
        if support is None:
            self.support_valid = False
            self.surface_support_point = np.array(self.position, dtype=np.float32)
            self.surface_support_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.surface_support_face_index = -1
        else:
            self.support_valid = True
            self.surface_support_point = np.asarray(support["point"], dtype=np.float32)
            self.surface_support_normal = np.asarray(support["normal"], dtype=np.float32)
            self.surface_support_face_index = int(support["face_index"])

        surface_forward = self._project_onto_plane(
            self.direction_world,
            self.surface_support_normal,
        )
        if surface_forward is None:
            surface_forward = self._normalize_xz(self.direction_world)
        if surface_forward is None:
            surface_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.surface_forward = np.asarray(surface_forward, dtype=np.float32)

    def _surface_right_vector(self) -> np.ndarray:
        flat_forward = self._normalize_xz(self.direction_world)
        if flat_forward is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(flat_forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        normalized_right = self._normalize(right)
        if normalized_right is None:
            normalized_right = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normalized_right

    def _cast_wall_hit(self, origin, direction, max_distance: float):
        origin = np.asarray(origin, dtype=np.float32)
        direction = self._normalize(direction)
        if direction is None or max_distance <= 1e-6:
            return None
        cast_origin = origin + direction * self.SURFACE_WALL_CAST_EPS
        hit = self._batch_ray_hits(
            self.wall_ray_finder,
            np.asarray([cast_origin], dtype=np.float32),
            np.asarray([direction], dtype=np.float32),
        )[0]
        if hit is None:
            return None
        distance = float(hit["distance"]) + self.SURFACE_WALL_CAST_EPS
        if distance <= (float(max_distance) + self.SURFACE_TRAVERSAL_EPS):
            return dict(
                position=np.asarray(hit["position"], dtype=np.float32),
                distance=distance,
                face_index=int(hit["face_index"]),
            )
        return None

    def _face_tangent_direction(self, face_index: int, direction) -> np.ndarray | None:
        if face_index < 0 or face_index >= len(self.road_traversal_mesh.faces):
            return None
        face_normal = self.road_traversal_mesh.face_normals[int(face_index)]
        tangent_dir = self._project_onto_plane(direction, face_normal)
        if tangent_dir is not None:
            return tangent_dir
        return self._project_onto_plane(self.surface_forward, face_normal)

    def _triangle_exit(self, face_index: int, point, direction) -> tuple[float, np.ndarray, list[int]] | None:
        if face_index < 0 or face_index >= len(self.road_traversal_mesh.faces):
            return None

        triangle = np.asarray(
            self.road_traversal_mesh.triangles[int(face_index)],
            dtype=np.float64,
        )
        point = np.asarray(point, dtype=np.float64)
        direction = self._normalize(direction)
        if direction is None:
            return None
        direction = np.asarray(direction, dtype=np.float64)

        bary0 = points_to_barycentric(
            triangle.reshape(1, 3, 3),
            point.reshape(1, 3),
        )[0]
        bary0 = np.asarray(bary0, dtype=np.float64)
        bary0[np.abs(bary0) < self.SURFACE_TRAVERSAL_EPS] = 0.0

        bary1 = points_to_barycentric(
            triangle.reshape(1, 3, 3),
            (point + direction).reshape(1, 3),
        )[0]
        bary1 = np.asarray(bary1, dtype=np.float64)
        deriv = bary1 - bary0

        candidates: list[tuple[float, int]] = []
        for bary_index, deriv_value in enumerate(deriv):
            if deriv_value < -self.SURFACE_TRAVERSAL_EPS:
                t = -float(bary0[bary_index]) / float(deriv_value)
                if t > self.SURFACE_TRAVERSAL_EPS:
                    candidates.append((t, bary_index))

        if not candidates:
            return None

        exit_distance = min(candidate[0] for candidate in candidates)
        crossed_bary_indices = [
            bary_index
            for t, bary_index in candidates
            if abs(float(t) - float(exit_distance)) <= 1e-5
        ]
        exit_point = point + direction * float(exit_distance)
        return float(exit_distance), np.asarray(exit_point, dtype=np.float32), crossed_bary_indices

    def _pick_neighbor_face(
        self,
        face_index: int,
        crossed_bary_indices: list[int],
        direction,
    ) -> int:
        if face_index < 0 or face_index >= len(self.road_face_neighbors):
            return -1

        best_neighbor = -1
        best_score = -np.inf
        for bary_index in crossed_bary_indices:
            if bary_index < 0 or bary_index >= 3:
                continue
            neighbor = int(self.road_face_neighbors[int(face_index), int(bary_index)])
            if neighbor < 0:
                continue
            neighbor_dir = self._face_tangent_direction(neighbor, direction)
            if neighbor_dir is None:
                continue
            score = float(np.dot(neighbor_dir, self._normalize(direction)))
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        return best_neighbor
    
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

        direction_world = np.array([data['dx'], data['dy'], data['dz']], dtype=np.float32)
        direction_world_norm = np.linalg.norm(direction_world)
        if direction_world_norm > 1e-6:
            self.direction_world = direction_world / direction_world_norm

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
        if self.vertical_mode:
            self._update_surface_state()
            data["forward_y"] = float(self.direction_world[1])
            data["support_normal_y"] = float(np.clip(self.surface_support_normal[1], -1.0, 1.0))
            data["cross_slope"] = float(
                np.clip(
                    np.dot(self.surface_support_normal, self._surface_right_vector()),
                    -1.0,
                    1.0,
                )
            )
            data["support_valid"] = float(self.support_valid)
        else:
            data["forward_y"] = 0.0
            data["support_normal_y"] = 1.0
            data["cross_slope"] = 0.0
            data["support_valid"] = 0.0
        self.generate_laser_directions(Car.ANGLE)
        self.find_closest_intersections()
        data["laser_elevation_rates"] = self.laser_elevation_rates.copy()
        data["laser_surface_lengths"] = self.laser_surface_lengths.copy()
        data["laser_elevation_deltas"] = self.laser_elevation_deltas.copy()
        data["ray_debug_modes"] = list(self.ray_debug_modes)
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

        values = struct.unpack(f"<{Car.PACKET_FLOAT_COUNT}f", packet)
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
            "ground_contact_fl",
            "ground_contact_fr",
            "ground_contact_rl",
            "ground_contact_rr",
            "ground_material_fl",
            "ground_material_fr",
            "ground_material_rl",
            "ground_material_rr",
            "icing_fl",
            "icing_fr",
            "icing_rl",
            "icing_rr",
            "dirt_fl",
            "dirt_fr",
            "dirt_rl",
            "dirt_rr",
            "wetness",
        ]
        data = dict(zip(keys, values))
        data["y"] += 0.2
        return data

    def visualize_rays(self, scene: trimesh.Scene):
        color_by_mode = {
            "flat_wall": [0, 0, 255],
            "flat_open": [100, 100, 255],
            "surface_wall": [255, 64, 64],
            "surface_wall_fallback": [255, 128, 64],
            "surface_open": [0, 200, 255],
            "surface_probe_miss": [255, 220, 64],
        }

        for i, ray_end in enumerate(self.intersections):
            scene.delete_geometry(f"ray{i}")
            ray_origin = self.position
            ray_path = self.ray_paths[i] if i < len(self.ray_paths) else [ray_origin, ray_end]
            if len(ray_path) < 2:
                ray_path = [ray_origin, ray_end]
            ray_color = color_by_mode.get(self.ray_debug_modes[i], [0, 0, 255])
            ray_geometry = trimesh.load_path(np.asarray(ray_path, dtype=np.float32), colors=[ray_color])
            scene.add_geometry(ray_geometry, node_name=f"ray{i}")
        
        # for i, next_point in enumerate(self.next_points):
        #    scene.delete_geometry(f"path{i}")
        #    line_geometry = trimesh.load_path([self.position, next_point], colors=[[255, 0, 255]])
        #    
        #    scene.add_geometry(line_geometry, node_name=f"path{i}")

        scene.delete_geometry("support_normal")
        support_origin = self.surface_support_point
        support_end = support_origin + self.surface_support_normal * 8.0
        support_geometry = trimesh.load_path(
            np.asarray([support_origin, support_end], dtype=np.float32),
            colors=[[255, 0, 255]],
        )
        scene.add_geometry(support_geometry, node_name="support_normal")

    
        
    
    def _find_flat_intersections(self):
        ray_origin = self.position
        ray_origins = np.repeat(np.asarray([ray_origin], dtype=np.float32), Car.NUM_LASERS, axis=0)
        hits = self._batch_ray_hits(
            self.wall_ray_finder,
            ray_origins,
            np.asarray(self.rays_directions, dtype=np.float32),
        )

        for i in range(Car.NUM_LASERS):
            hit = hits[i] if i < len(hits) else None
            if hit is None:
                distance = Car.LASER_MAX_DISTANCE
                hit_position = ray_origin + self.rays_directions[i] * distance
                mode = "flat_open"
            else:
                distance = float(np.linalg.norm(np.asarray(self.position, dtype=np.float32) - hit["position"]))
                hit_position = hit["position"]
                mode = "flat_wall"
            self.intersections[i] = np.asarray(hit_position, dtype=np.float32)
            self.distances[i] = distance
            self.ray_paths[i] = [
                np.asarray(ray_origin, dtype=np.float32),
                np.asarray(hit_position, dtype=np.float32),
            ]
            self.ray_debug_modes[i] = mode
            self.laser_elevation_rates[i] = 0.0
            self.laser_elevation_deltas[i] = 0.0
            self.laser_surface_lengths[i] = float(distance)

    def _find_surface_intersections(self):
        if not self.support_valid:
            self._find_flat_intersections()
            return

        start_point = np.asarray(self.surface_support_point, dtype=np.float32)
        start_normal = np.asarray(self.surface_support_normal, dtype=np.float32)
        start_face_index = int(self.surface_support_face_index)
        if start_face_index < 0 or start_face_index >= len(self.road_traversal_mesh.faces):
            self._find_flat_intersections()
            return

        lift_start = start_point + start_normal * self.surface_ray_lift
        self.ray_paths = [
            [np.asarray(self.position, dtype=np.float32), np.asarray(lift_start, dtype=np.float32)]
            for _ in range(Car.NUM_LASERS)
        ]

        for idx in range(Car.NUM_LASERS):
            current_face_index = start_face_index
            current_point = np.asarray(start_point, dtype=np.float32)
            total_distance = 0.0
            final_mode = "surface_open"
            final_point = np.asarray(lift_start, dtype=np.float32)

            max_face_hops = max(1, len(self.road_traversal_mesh.faces))
            for _ in range(max_face_hops):
                remaining = Car.LASER_MAX_DISTANCE - float(total_distance)
                if remaining <= self.SURFACE_TRAVERSAL_EPS:
                    final_mode = "surface_open"
                    final_point = (
                        current_point
                        + np.asarray(
                            self.road_traversal_mesh.face_normals[int(current_face_index)],
                            dtype=np.float32,
                        )
                        * self.surface_ray_lift
                    )
                    total_distance = Car.LASER_MAX_DISTANCE
                    break

                tangent_dir = self._face_tangent_direction(
                    current_face_index,
                    self.rays_directions[idx],
                )
                if tangent_dir is None:
                    final_mode = "surface_probe_miss"
                    final_point = np.asarray(current_point, dtype=np.float32)
                    break

                exit_data = self._triangle_exit(
                    current_face_index,
                    current_point,
                    tangent_dir,
                )
                if exit_data is None:
                    wall_hit = self._cast_wall_hit(
                        origin=current_point,
                        direction=tangent_dir,
                        max_distance=remaining,
                    )
                    if wall_hit is not None:
                        total_distance += float(wall_hit["distance"])
                        final_point = np.asarray(wall_hit["position"], dtype=np.float32)
                        final_mode = "surface_wall"
                        self.ray_paths[idx].append(final_point)
                    else:
                        total_distance = Car.LASER_MAX_DISTANCE
                        final_point = current_point + np.asarray(tangent_dir, dtype=np.float32) * remaining
                        final_mode = "surface_open"
                        self.ray_paths[idx].append(final_point)
                    break

                edge_distance, exit_point, crossed_bary_indices = exit_data
                if edge_distance > remaining:
                    wall_hit = self._cast_wall_hit(
                        origin=current_point,
                        direction=tangent_dir,
                        max_distance=remaining,
                    )
                    if wall_hit is not None:
                        total_distance += float(wall_hit["distance"])
                        final_point = np.asarray(wall_hit["position"], dtype=np.float32)
                        final_mode = "surface_wall"
                        self.ray_paths[idx].append(final_point)
                    else:
                        total_distance = Car.LASER_MAX_DISTANCE
                        final_point = current_point + np.asarray(tangent_dir, dtype=np.float32) * remaining
                        final_mode = "surface_open"
                        self.ray_paths[idx].append(final_point)
                    break

                wall_hit = self._cast_wall_hit(
                    origin=current_point,
                    direction=tangent_dir,
                    max_distance=edge_distance,
                )
                if wall_hit is not None:
                    total_distance += float(wall_hit["distance"])
                    final_point = np.asarray(wall_hit["position"], dtype=np.float32)
                    final_mode = "surface_wall"
                    self.ray_paths[idx].append(final_point)
                    break

                total_distance += float(edge_distance)
                current_normal = np.asarray(
                    self.road_traversal_mesh.face_normals[int(current_face_index)],
                    dtype=np.float32,
                )
                self.ray_paths[idx].append(
                    np.asarray(exit_point, dtype=np.float32) + current_normal * self.surface_ray_lift
                )

                next_face_index = self._pick_neighbor_face(
                    current_face_index,
                    crossed_bary_indices,
                    tangent_dir,
                )
                if next_face_index < 0:
                    final_mode = "surface_open"
                    final_point = np.asarray(exit_point, dtype=np.float32) + current_normal * self.surface_ray_lift
                    break

                next_tangent_dir = self._face_tangent_direction(
                    next_face_index,
                    tangent_dir,
                )
                if next_tangent_dir is None:
                    final_mode = "surface_probe_miss"
                    final_point = np.asarray(exit_point, dtype=np.float32)
                    break

                next_normal = np.asarray(
                    self.road_traversal_mesh.face_normals[int(next_face_index)],
                    dtype=np.float32,
                )
                current_point = np.asarray(exit_point, dtype=np.float32) + np.asarray(
                    next_tangent_dir,
                    dtype=np.float32,
                ) * self.SURFACE_TRAVERSAL_EPS
                current_face_index = int(next_face_index)
                final_point = current_point + next_normal * self.surface_ray_lift
                final_mode = "surface_open"
            else:
                final_mode = "surface_probe_miss"
                final_point = np.asarray(current_point, dtype=np.float32)

            clamped_distance = min(float(total_distance), Car.LASER_MAX_DISTANCE)
            self.intersections[idx] = np.asarray(final_point, dtype=np.float32)
            self.distances[idx] = clamped_distance
            self.ray_debug_modes[idx] = final_mode
            self.laser_surface_lengths[idx] = clamped_distance
            self.laser_elevation_deltas[idx] = float(final_point[1] - start_point[1])
            if clamped_distance > 1e-6:
                self.laser_elevation_rates[idx] = float(
                    self.laser_elevation_deltas[idx] / clamped_distance
                )
            else:
                self.laser_elevation_rates[idx] = 0.0
            if len(self.ray_paths[idx]) < 2:
                self.ray_paths[idx] = [
                    np.asarray(self.position, dtype=np.float32),
                    np.asarray(final_point, dtype=np.float32),
                ]

    def find_closest_intersections(self):
        if self.vertical_mode:
            self._find_surface_intersections()
        else:
            self._find_flat_intersections()
                
    def reset(self):
        self.position = np.array([0, 0, 0])
        self.direction = np.array([1, 0, 0])
        self.direction_world = np.array([1, 0, 0], dtype=np.float32)
        self.path_tile_index = 0
        self.speed = 0
        self.side_speed = 0
        self.distance_traveled = 0
        self.wheel_slips = np.zeros(4, dtype=np.float32)
        self.surface_support_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.surface_support_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.surface_support_face_index = -1
        self.surface_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.support_valid = False

        self.last_position = self.position
        self.last_direction = self.direction

        self.distances = [Car.LASER_MAX_DISTANCE for _ in range(Car.NUM_LASERS)]
        self.rays_directions = [[1, 0, 0] for _ in range(Car.NUM_LASERS)]
        self.intersections = [[0, 0, 0] for _ in range(Car.NUM_LASERS)]
        self.ray_paths = [[np.array([0.0, 0.0, 0.0], dtype=np.float32)] for _ in range(Car.NUM_LASERS)]
        self.ray_debug_modes = ["flat_open" for _ in range(Car.NUM_LASERS)]
        self.laser_elevation_rates = np.zeros(Car.NUM_LASERS, dtype=np.float32)
        self.laser_elevation_deltas = np.zeros(Car.NUM_LASERS, dtype=np.float32)
        self.laser_surface_lengths = np.zeros(Car.NUM_LASERS, dtype=np.float32)

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

        if self.vertical_mode:
            rotation_axis = np.asarray(self.surface_support_normal, dtype=np.float32)
            front_direction = np.asarray(self.surface_forward, dtype=np.float32)
        else:
            rotation_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            front_direction = np.asarray(self.direction, dtype=np.float32)

        # Generate unit vectors for laser directions
        laser_directions = []
        for i in range(Car.NUM_LASERS):
            # Calculate the angle offset for this laser
            angle_offset = i * angular_spacing - angle_range_radians / 2

            # Rotate the front direction vector by the angle offset to get the laser direction
            rotation_matrix = trimesh.transformations.rotation_matrix(angle_offset, rotation_axis)
            laser_direction = np.dot(rotation_matrix[:3, :3], front_direction)
            if self.vertical_mode:
                projected_direction = self._project_onto_plane(
                    laser_direction,
                    self.surface_support_normal,
                )
                if projected_direction is not None:
                    laser_direction = projected_direction

            # Add the laser direction to the list
            laser_directions.append(laser_direction)
        
        self.rays_directions = laser_directions
