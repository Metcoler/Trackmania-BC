import trimesh
import numpy as np
import trimesh.visual.color
import time
from pathlib import Path
from SurfaceTypes import traction_for_surface_prefix

MAP_BLOCK_SIZE = 32
MAP_GROUND_LEVEL = -4*(MAP_BLOCK_SIZE // 2)

class MapBlock:
    SLOPE_SENSOR_CURTAIN_HEIGHT = 6.0

    SURFACE_PREFIXES = (
        "RoadTech",
        "RoadDirt",
        "PlatformIce",
        "PlatformGrass",
        "PlatformPlastic",
        "PlatformDirt",
    )

    direction_dictionary = {
        "N": 0,
        "E": 270,
        "S": 180,
        "W": 90
    }

    block_colors = {
        "Start": [0, 255, 0],
        "Finish": [255, 0, 0],
        "Checkpoint": [0, 0, 255],
    }

    in_out_tiles = {
        "Start": (np.array([0, 0, 0]), np.array([0, 0, 0])),
        "Finish": (np.array([0, 0, 0]), np.array([0, 0, 0])),
        "Checkpoint": (np.array([0, 0, 0]), np.array([0, 0, 0])),
        "Straight": (np.array([0, 0, 0]), np.array([0, 0, 0])),
        "SlopeBase": (np.array([0, 0, 0]), np.array([0, 1, 0])),
        "SlopeBase2x1": (np.array([0, 0, 0]), np.array([0, 1, 1])),
        "Curve": (np.array([0, 0, 0]), np.array([1, 0, 1])),
        "SlopeBaseCurve2Right": (np.array([0, 0, 0]), np.array([1, 1, 1])),
        "SlopeBaseCurve2Left": (np.array([1, 0, 0]), np.array([0, 1, 1])),
        "Slope2BaseCurve2Right": (np.array([0, 0, 0]), np.array([1, 2, 1])),
        "Slope2BaseCurve2Left": (np.array([1, 0, 0]), np.array([0, 2, 1])),
    }

    in_out_directions = {
        "Start": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Finish": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Checkpoint": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Straight": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "SlopeBase": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "SlopeBase2x1": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Curve": (np.array([1, 0, 0]), np.array([0, 0, 1])),
        "SlopeBaseCurve2Right": (np.array([1, 0, 0]), np.array([0, 0, 1])),
        "SlopeBaseCurve2Left": (np.array([-1, 0, 0]), np.array([0, 0, 1])),
        "Slope2BaseCurve2Right": (np.array([1, 0, 0]), np.array([0, 0, 1])),
        "Slope2BaseCurve2Left": (np.array([-1, 0, 0]), np.array([0, 0, 1])),
    }

    SPECIAL_BLOCKS = {
        "SlopeBase2x1": {
            "block_size": 2,
            "footprint": (1, 2),
        },
        "SlopeBaseCurve2Right": {
            "block_size": 2,
            "footprint": (2, 2),
        },
        "SlopeBaseCurve2Left": {
            "block_size": 2,
            "footprint": (2, 2),
        },
        "Slope2BaseCurve2Right": {
            "block_size": 2,
            "footprint": (2, 2),
        },
        "Slope2BaseCurve2Left": {
            "block_size": 2,
            "footprint": (2, 2),
        },
    }

    EXACT_TILE_OFFSET_BLOCKS = set(SPECIAL_BLOCKS)
    
    @classmethod
    def _strip_surface_prefix(cls, block_name: str) -> tuple[str, str]:
        for prefix in cls.SURFACE_PREFIXES:
            if block_name.startswith(prefix):
                return prefix, block_name[len(prefix):]
        return "Unknown", block_name

    @classmethod
    def resolve_block_name(cls, raw_name: str) -> tuple[str, str, str, int, tuple[int, int]]:
        surface, shape_name = cls._strip_surface_prefix(raw_name)

        if shape_name == "Base":
            shape_name = "Straight"
        elif shape_name == "Slope2Base":
            shape_name = "SlopeBase2"

        if shape_name in cls.SPECIAL_BLOCKS:
            block_info = cls.SPECIAL_BLOCKS[shape_name]
            block_size = int(block_info["block_size"])
            footprint = tuple(int(value) for value in block_info["footprint"])
            mesh_name = f"RoadTech{shape_name}"
            mesh_path = Path("Meshes") / f"{mesh_name}.obj"
            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"Could not resolve mesh for block '{raw_name}'. "
                    f"Tried '{mesh_path}'."
                )
            return mesh_name, shape_name, surface, block_size, footprint

        block_size = 1
        semantic_name = shape_name
        if semantic_name[-1:].isdigit():
            block_size = int(semantic_name[-1])
            semantic_name = semantic_name[:-1]

        if semantic_name == "Base":
            semantic_name = "Straight"

        if semantic_name in {"Start", "Finish", "Checkpoint"}:
            mesh_name = f"RoadTech{semantic_name}"
        elif semantic_name == "Straight":
            mesh_name = "RoadTechStraight"
        elif semantic_name == "Curve":
            mesh_name = f"RoadTechCurve{block_size}"
        elif semantic_name == "SlopeBase":
            mesh_name = "RoadTechSlopeBase" if block_size == 1 else f"RoadTechSlopeBase{block_size}"
        else:
            mesh_name = raw_name

        mesh_path = Path("Meshes") / f"{mesh_name}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Could not resolve mesh for block '{raw_name}'. "
                f"Tried '{mesh_path}'."
            )

        return mesh_name, semantic_name, surface, block_size, (block_size, block_size)

    def __init__(self, name: str, logical_position: tuple[int], direction: str) -> None:

        self.logical_position = np.array(logical_position)
        self.position = np.array([logical_position[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + logical_position[1] * MAP_BLOCK_SIZE // 4, logical_position[2] * MAP_BLOCK_SIZE])
        self.raw_name = name
        mesh_name, semantic_name, surface_name, block_size, footprint_tiles = self.resolve_block_name(name)
        self.mesh_name = mesh_name
        self.surface_name = surface_name
        
        self.mesh = trimesh.load(f"Meshes/{mesh_name}.obj", force="mesh", process=True)
        
        # get block logical size
        self.block_size = block_size
        self.footprint_tiles = tuple(int(value) for value in footprint_tiles)
        name = semantic_name
        self.name = semantic_name 

        self.color = [np.random.randint(0, 255) for _ in range(3)]
        
        # update position
        angle = self.direction_dictionary[direction]
        footprint_center = [
            self.footprint_tiles[0] * MAP_BLOCK_SIZE * 0.5,
            0,
            self.footprint_tiles[1] * MAP_BLOCK_SIZE * 0.5,
        ]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], footprint_center)
        self.mesh.apply_transform(rotation_matrix)
        self.mesh.apply_translation(self.position + self.non_square_rotation_correction(direction)) 
        self.bounds = np.asarray(self.mesh.bounds, dtype=np.float32)
         
        # center
        self.center_point = self.position + np.array(
            [
                self.footprint_tiles[0] * MAP_BLOCK_SIZE * 0.5,
                0,
                self.footprint_tiles[1] * MAP_BLOCK_SIZE * 0.5,
            ]
        )
        if self.footprint_tiles != (1, 1):
            self.center_point = self.mesh.centroid 
        self.center_point = list(self.center_point)
        self.center_point[1] = self.position[1]
        self.center_point = np.array(self.center_point)
        
        self.calculate_in_out_points(name, direction)
        self.calculate_in_out_vectors(name, direction)        
        self.split_mesh_into_walls_and_road()
        self.fit_road_plane()
        self.build_sensor_walls_mesh()
        self.generate_mesh_points()

        # color the mesh
        if name in MapBlock.block_colors:
            self.set_color(MapBlock.block_colors[name])

    def non_square_rotation_correction(self, direction: str) -> np.ndarray:
        if self.footprint_tiles[0] == self.footprint_tiles[1]:
            return np.zeros(3, dtype=np.float64)

        # The 2x1 slope mesh is authored as a one-tile-wide/two-tile-long north
        # block. After a 90-degree rotation around its center, the long axis sits
        # half a tile off the Trackmania grid. Shift it back onto full grid cells
        # while keeping the logical in/out tile convention unchanged.
        half_tile_delta = 0.5 * MAP_BLOCK_SIZE * (self.footprint_tiles[1] - self.footprint_tiles[0])
        if self.name == "SlopeBase2x1" and direction in {"E", "W"}:
            return np.array([half_tile_delta, 0.0, -half_tile_delta], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)


    def calculate_in_out_points(self, name, direction):
        # in and out points
        pre_rotated_tile_offsets = False
        if name == "SlopeBase2x1":
            oriented_tiles = {
                "N": (np.array([0, 0, 0]), np.array([0, 1, 1])),
                "E": (np.array([1, 0, 0]), np.array([0, 1, 0])),
                "S": (np.array([0, 0, 1]), np.array([0, 1, 0])),
                "W": (np.array([0, 0, 0]), np.array([1, 1, 0])),
            }
            self.in_tile, self.out_tile = oriented_tiles[direction]
            pre_rotated_tile_offsets = True
        else:
            self.in_tile, self.out_tile = self.in_out_tiles[name]

        if name in self.EXACT_TILE_OFFSET_BLOCKS:
            self.in_tile = list(self.in_tile) + [1]
            self.out_tile = list(self.out_tile) + [1]
        elif "Slope" not in name:
            self.in_tile = list(self.in_tile * (self.block_size-1)) + [1]
            self.out_tile = list(self.out_tile * (self.block_size-1)) + [1]
        else:
            self.in_tile = list(self.in_tile * (self.block_size)) + [1]
            self.out_tile = list(self.out_tile * (self.block_size)) + [1]

        angle = self.direction_dictionary[direction]
        tile_rotation_center = [
            (self.footprint_tiles[0] - 1) * 0.5,
            0,
            (self.footprint_tiles[1] - 1) * 0.5,
        ]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], tile_rotation_center)

        if pre_rotated_tile_offsets:
            self.in_tile = np.asarray(self.in_tile[:3], dtype=np.float64) + self.logical_position
            self.out_tile = np.asarray(self.out_tile[:3], dtype=np.float64) + self.logical_position
        else:
            self.in_tile = np.round(np.array(np.dot(rotation_matrix, self.in_tile)[:3])) + self.logical_position
            self.out_tile = np.round(np.array(np.dot(rotation_matrix, self.out_tile)[:3])) + self.logical_position

        self.in_point = np.array([16+self.in_tile[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + self.in_tile[1]*MAP_BLOCK_SIZE//4, 16+self.in_tile[2] * MAP_BLOCK_SIZE])
        self.out_point = np.array([16+self.out_tile[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + self.out_tile[1]*MAP_BLOCK_SIZE//4, 16+self.out_tile[2] * MAP_BLOCK_SIZE])
    
    def calculate_in_out_vectors(self, name, direction):
        in_vector, out_vector = self.in_out_directions[name]
        in_vector = list(in_vector) + [0]
        out_vector = list(out_vector) + [0]

        angle = self.direction_dictionary[direction]

        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [0, 0, 0])
        self.in_vector = np.round(np.dot(rotation_matrix, in_vector)[:3])
        self.out_vector = np.round(np.dot(rotation_matrix, out_vector)[:3])


    def swap_in_out(self):
        self.in_tile, self.out_tile = self.out_tile, self.in_tile
        self.in_point, self.out_point = self.out_point, self.in_point
        self.in_vector, self.out_vector = -self.out_vector, -self.in_vector
        self.generate_mesh_points()
    
    def generate_mesh_points(self):
        ## center
        self.center_point_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(self.center_point))
        self.center_point_mesh.visual.vertex_colors = [255, 0, 0]

        # origin point
        self.origin_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(self.position))
        self.origin_mesh.visual.vertex_colors = self.color

        # in point
        self.in_point_mesh = trimesh.creation.box(extents=[1, 1, 2], transform=trimesh.transformations.translation_matrix(self.in_point))
        self.in_point_mesh.visual.vertex_colors = self.color

        # out point
        self.out_point_mesh = trimesh.creation.box(extents=[2, 1, 1], transform=trimesh.transformations.translation_matrix(self.out_point))
        self.out_point_mesh.visual.vertex_colors = self.color

        # in vector
        ray_origin = self.in_point
        ray_end = self.in_point + self.in_vector * 10
        self.in_vector_mesh = trimesh.load_path([ray_origin, ray_end], colors=[[0, 255, 0]])
    
        # out vector
        ray_origin = self.out_point
        ray_end = self.out_point + self.out_vector * 10
        self.out_vector_mesh = trimesh.load_path([ray_origin, ray_end], colors=[[255, 0, 0]])

    def split_mesh_into_walls_and_road(self):
        normals = self.mesh.face_normals

        # Find indices of triangles with normals not pointing 
        walls_indices = []
        road_indices = []
        for i, normal in enumerate(normals):
            normalized = normal / np.linalg.norm(normal)
            if np.abs(np.dot(normalized, [0, 1, 0])) < 0.11:
                walls_indices.append(i)
            else:
                road_indices.append(i)


        # Walls - Create a new mesh with faces not pointing in the y-axis direction
        self.walls_mesh = trimesh.Trimesh(vertices=self.mesh.vertices,
                                    faces=self.mesh.faces[walls_indices])
        self.walls_mesh.visual.vertex_colors = [50, 50, 50]
        self.walls_mesh.remove_unreferenced_vertices()



        # Road - Create a new mesh with faces pointing in the y-axis direction
        self.road_mesh = trimesh.Trimesh(vertices=self.mesh.vertices,
                                    faces=self.mesh.faces[road_indices])
        self.road_mesh.visual.vertex_colors = [95, 145, 95, 255]
        self.road_mesh.visual.face_colors = [95, 145, 95, 255]
        self.road_mesh.remove_unreferenced_vertices()

    def fit_road_plane(self):
        if len(self.road_mesh.faces) == 0:
            self.road_plane = (0.0, 0.0, float(self.position[1]))
            self.road_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return

        normals = np.asarray(self.road_mesh.face_normals, dtype=np.float64)
        mask = normals[:, 1] > 0.5
        triangles = np.asarray(self.road_mesh.triangles[mask], dtype=np.float64)
        if triangles.size == 0:
            triangles = np.asarray(self.road_mesh.triangles, dtype=np.float64)
        points = triangles.reshape(-1, 3)
        if points.shape[0] < 3:
            height = float(self.position[1])
            self.road_plane = (0.0, 0.0, height)
            self.road_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return

        design = np.column_stack([points[:, 0], points[:, 2], np.ones(points.shape[0])])
        coeffs, *_ = np.linalg.lstsq(design, points[:, 1], rcond=None)
        a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        self.road_plane = (a, b, c)
        normal = np.array([-a, 1.0, -b], dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-6:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            normal = normal / normal_norm
        self.road_normal = normal

    def build_sensor_walls_mesh(self):
        # Keep the real wall mesh intact and add simple vertical side curtains only on
        # slope blocks. The curtain bottom follows the road edge exactly; only the
        # top helper edge is lifted, so the slope start/end heights remain unchanged.
        self.sensor_walls_mesh = self.walls_mesh.copy()
        curtain_mesh = self.build_slope_sensor_curtains()
        if curtain_mesh is not None and len(curtain_mesh.faces) > 0:
            self.sensor_walls_mesh += curtain_mesh
        self.sensor_wall_ray_finder = None
        if len(self.sensor_walls_mesh.faces) > 0:
            self.sensor_wall_ray_finder = trimesh.ray.ray_triangle.RayMeshIntersector(
                self.sensor_walls_mesh
            )

    def build_slope_sensor_curtains(self):
        if "Slope" not in self.name or len(self.road_mesh.faces) == 0:
            return None

        edge_owner: dict[tuple[int, int], int] = {}
        boundary_edges: list[tuple[int, int]] = []
        for face in np.asarray(self.road_mesh.faces, dtype=np.int64):
            for vertex_a, vertex_b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((int(vertex_a), int(vertex_b))))
                if edge in edge_owner:
                    edge_owner[edge] += 1
                else:
                    edge_owner[edge] = 1
        for edge, owner_count in edge_owner.items():
            if owner_count == 1:
                boundary_edges.append(edge)

        vertices: list[np.ndarray] = []
        faces: list[list[int]] = []
        for vertex_a, vertex_b in boundary_edges:
            point_a = np.asarray(self.road_mesh.vertices[int(vertex_a)], dtype=np.float64)
            point_b = np.asarray(self.road_mesh.vertices[int(vertex_b)], dtype=np.float64)
            edge_xz = np.asarray([point_b[0] - point_a[0], point_b[2] - point_a[2]], dtype=np.float64)
            edge_norm = float(np.linalg.norm(edge_xz))
            if edge_norm <= 1e-6:
                continue
            edge_xz /= edge_norm

            # Entry and exit edges are cross-track openings. Keep those open so
            # grid traversal can move into connected blocks, and only add helper
            # wall curtains along side boundaries. On curve-slope blocks the side
            # boundary is a polyline, so a simple "parallel to slope direction"
            # test is not sufficient.
            if self._is_open_road_boundary_edge(point_a, point_b, edge_xz):
                continue

            half_height = self.SLOPE_SENSOR_CURTAIN_HEIGHT * 0.5
            bottom_a = point_a + np.array([0.0, -half_height, 0.0], dtype=np.float64)
            bottom_b = point_b + np.array([0.0, -half_height, 0.0], dtype=np.float64)
            top_a = point_a + np.array([0.0, half_height, 0.0], dtype=np.float64)
            top_b = point_b + np.array([0.0, half_height, 0.0], dtype=np.float64)
            base_index = len(vertices)
            vertices.extend([bottom_a, bottom_b, top_b, top_a])
            faces.append([base_index, base_index + 1, base_index + 2])
            faces.append([base_index, base_index + 2, base_index + 3])

        if not vertices:
            return None

        curtain_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        curtain_mesh.visual.face_colors = [255, 128, 0, 180]
        return curtain_mesh

    def _is_open_road_boundary_edge(self, point_a, point_b, edge_xz: np.ndarray) -> bool:
        midpoint_xz = np.asarray(
            [
                0.5 * (float(point_a[0]) + float(point_b[0])),
                0.5 * (float(point_a[2]) + float(point_b[2])),
            ],
            dtype=np.float64,
        )

        for opening_point, opening_vector in (
            (self.in_point, self.in_vector),
            (self.out_point, self.out_vector),
        ):
            direction_xz = np.asarray([opening_vector[0], opening_vector[2]], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction_xz))
            if direction_norm <= 1e-6:
                continue
            direction_xz /= direction_norm

            # Opening edges run across the track, so they are roughly
            # perpendicular to the local travel direction.
            if abs(float(np.dot(edge_xz, direction_xz))) > 0.45:
                continue

            rel = midpoint_xz - np.asarray([opening_point[0], opening_point[2]], dtype=np.float64)
            longitudinal_distance = abs(float(np.dot(rel, direction_xz)))
            lateral_vector = rel - float(np.dot(rel, direction_xz)) * direction_xz
            lateral_distance = float(np.linalg.norm(lateral_vector))
            # The in/out point is stored at the center of the logical path tile,
            # while the real mesh opening is on that tile boundary, roughly half
            # a tile away. Keep that seam open, but do not swallow the curved
            # side polyline where individual segments can point in many normals.
            if (
                longitudinal_distance <= MAP_BLOCK_SIZE * 0.65
                and lateral_distance <= MAP_BLOCK_SIZE * 0.55
            ):
                return True

        return False

    def road_height_at(self, x: float, z: float) -> float:
        a, b, c = self.road_plane
        return float(a * float(x) + b * float(z) + c)

    def road_direction_for_xz(self, direction_xz) -> np.ndarray:
        direction_xz = np.asarray(direction_xz, dtype=np.float32)
        norm = float(np.linalg.norm(direction_xz))
        if norm <= 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        direction_xz = direction_xz / norm
        a, b, _ = self.road_plane
        direction = np.array(
            [direction_xz[0], a * direction_xz[0] + b * direction_xz[1], direction_xz[1]],
            dtype=np.float32,
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return np.array([direction_xz[0], 0.0, direction_xz[1]], dtype=np.float32)
        return direction / direction_norm

    def road_point_at(self, x: float, z: float, lift: float = 0.0) -> np.ndarray:
        return np.array(
            [float(x), self.road_height_at(x, z) + float(lift), float(z)],
            dtype=np.float32,
        )

    def get_sensor_wall_ray_finder(self):
        return self.sensor_wall_ray_finder

    def get_sensor_walls_mesh(self):
        return self.sensor_walls_mesh

    def get_mesh(self):
        return self.mesh

    def get_bounds(self):
        return self.bounds
    
    def get_road_mesh(self):
        return self.road_mesh
    
    def get_walls_mesh(self):
        return self.walls_mesh
    
    def get_points_mesh(self):
        return self.in_point_mesh + self.out_point_mesh + self.center_point_mesh

    def set_color(self, color: list[int]):
        self.mesh.visual.vertex_colors = color
        self.mesh.visual.face_colors = color
        self.road_mesh.visual.vertex_colors = color

    
    def get_out_position_vector(self, input_logical_position, input_vector, try_index=0):
        if try_index > 1:
            raise Exception("Error: path is not continuous")
        if all(np.array(input_logical_position == self.in_tile)) and all(input_vector == self.in_vector):
            return self.out_tile, self.out_vector
        self.swap_in_out()
        return self.get_out_position_vector(input_logical_position, input_vector, try_index+1)
    
    def contains_in_out_tile(self, logical_position):
        return all(logical_position == self.in_tile) or all(logical_position == self.out_tile)
    
    def __repr__(self) -> str:
        return f"{self.name}{self.block_size if self.block_size > 1 else ''}: {self.logical_position}"

        
class Map:
    
    def __init__(self, map_name) -> None:
        print("Creating map...")
        self.map_name = map_name
        self.blocks: dict[tuple[int], MapBlock] = dict()
        self.start_logical_position = None
        self.end_logical_position = None 
        
        with open(f"Maps/ExportedBlocks/{map_name}.txt", 'r', encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '#':
                    continue
                block_name, logical_position, block_direction = line.split(";")

                logical_position = tuple(map(int, logical_position.split(",")))
                block = MapBlock(block_name, logical_position, block_direction[0])
                self.blocks[logical_position] = block

                if "Start" in block_name:
                    self.start_logical_position = logical_position
                elif "Finish" in block_name:
                    self.end_logical_position = logical_position

        self.generate_block_grid_index()
        self.construct_path()
        self.generate_map_mesh()
        self.generate_walls_mesh()
        self.generate_sensor_walls_mesh()
        self.generate_road_mesh()
        self.generate_road_traversal_data()
        self.generate_path_mesh()

    def tile_coordinate_to_point(logical_position, middle=True, dy=0):
        position = np.array([logical_position[0] * MAP_BLOCK_SIZE, dy + MAP_GROUND_LEVEL + logical_position[1] * MAP_BLOCK_SIZE // 4, logical_position[2] * MAP_BLOCK_SIZE])
        if middle:
            position += np.array([MAP_BLOCK_SIZE // 2, 0, MAP_BLOCK_SIZE // 2])
        return position

    @staticmethod
    def point_to_xz_cell(point) -> tuple[int, int]:
        point = np.asarray(point, dtype=np.float32)
        return int(np.floor(point[0] / MAP_BLOCK_SIZE)), int(np.floor(point[2] / MAP_BLOCK_SIZE))

    def generate_block_grid_index(self):
        self.block_grid_index: dict[tuple[int, int], list[MapBlock]] = {}
        eps = 1e-5
        for block in self.blocks.values():
            bounds = np.asarray(block.get_bounds(), dtype=np.float64)
            min_x = int(np.floor(bounds[0, 0] / MAP_BLOCK_SIZE))
            max_x = int(np.floor((bounds[1, 0] - eps) / MAP_BLOCK_SIZE))
            min_z = int(np.floor(bounds[0, 2] / MAP_BLOCK_SIZE))
            max_z = int(np.floor((bounds[1, 2] - eps) / MAP_BLOCK_SIZE))
            for cell_x in range(min_x, max_x + 1):
                for cell_z in range(min_z, max_z + 1):
                    self.block_grid_index.setdefault((cell_x, cell_z), []).append(block)

    def blocks_at_xz_cell(self, cell_x: int, cell_z: int) -> list[MapBlock]:
        return self.block_grid_index.get((int(cell_x), int(cell_z)), [])

    @staticmethod
    def _filter_blocks_containing_xz_point(
        blocks: list[MapBlock],
        point,
        margin: float = 0.05,
    ) -> list[MapBlock]:
        point = np.asarray(point, dtype=np.float32)
        containing_blocks: list[MapBlock] = []
        for block in blocks:
            bounds = block.get_bounds()
            if (
                bounds[0, 0] - margin <= point[0] <= bounds[1, 0] + margin
                and bounds[0, 2] - margin <= point[2] <= bounds[1, 2] + margin
            ):
                containing_blocks.append(block)
        return containing_blocks

    @staticmethod
    def _nearest_block_by_road_height(blocks: list[MapBlock], point) -> MapBlock | None:
        if not blocks:
            return None
        point = np.asarray(point, dtype=np.float32)
        reference_y = float(point[1])
        return min(
            blocks,
            key=lambda block: abs(block.road_height_at(float(point[0]), float(point[2])) - reference_y),
        )

    def find_block_for_point(self, point, fallback_block: MapBlock | None = None) -> MapBlock | None:
        point = np.asarray(point, dtype=np.float32)
        cell = self.point_to_xz_cell(point)
        candidates = self.blocks_at_xz_cell(*cell)
        containing_candidates = self._filter_blocks_containing_xz_point(candidates, point)
        if containing_candidates:
            candidates = containing_candidates
        if not candidates:
            if fallback_block is None:
                return None
            bounds = fallback_block.get_bounds()
            margin = 0.25
            inside_fallback_xz = (
                bounds[0, 0] - margin <= point[0] <= bounds[1, 0] + margin
                and bounds[0, 2] - margin <= point[2] <= bounds[1, 2] + margin
            )
            return fallback_block if inside_fallback_xz else None
        if fallback_block in candidates:
            fallback_height = fallback_block.road_height_at(float(point[0]), float(point[2]))
            if abs(fallback_height - float(point[1])) <= MAP_BLOCK_SIZE * 0.25:
                return fallback_block
        return self._nearest_block_by_road_height(candidates, point)

    def find_connected_block_for_point(self, point, from_block: MapBlock | None) -> MapBlock | None:
        point = np.asarray(point, dtype=np.float32)
        candidates = self.blocks_at_xz_cell(*self.point_to_xz_cell(point))
        containing_candidates = self._filter_blocks_containing_xz_point(candidates, point)
        if containing_candidates:
            candidates = containing_candidates
        if not candidates:
            return None
        if from_block in candidates:
            return from_block
        connected_candidates = [
            block
            for block in candidates
            if self.can_transition_between_blocks(from_block, block)
        ]
        return self._nearest_block_by_road_height(connected_candidates, point)

    def cell_has_stacked_layers(
        self,
        cell_x: int,
        cell_z: int,
        height_threshold: float = 2.0,
    ) -> bool:
        candidates = self.blocks_at_xz_cell(cell_x, cell_z)
        if len(candidates) < 2:
            return False
        sample_x = (int(cell_x) + 0.5) * MAP_BLOCK_SIZE
        sample_z = (int(cell_z) + 0.5) * MAP_BLOCK_SIZE
        heights = [
            block.road_height_at(sample_x, sample_z)
            for block in candidates
        ]
        return (max(heights) - min(heights)) > float(height_threshold)

        
    def construct_path(self):
        path = [self.start_logical_position]
        
        logical_position = tuple(self.start_logical_position)
        in_position = self.start_logical_position
        in_vector = self.blocks[logical_position].out_vector
        
        while self.blocks[logical_position].name != "Finish":

            out_position, out_vector = self.blocks[logical_position].get_out_position_vector(in_position, in_vector)
            next_in_position = out_position + out_vector
            
            for block in self.blocks.values():
                if block.contains_in_out_tile(next_in_position):
                
                    in_position = next_in_position
                    in_vector = out_vector
                    logical_position = tuple(block.logical_position)
                    path.append(logical_position)
                    break
            else:
                break
                #raise Exception("Error: path is not continuous")
        
        print("Path constructed...")
        self.block_path = path
        self.generate_block_path_transition_index()
        self.path_tiles = []
        self.path_instructions = []
        self.path_surface_instructions = []
        self.path_height_instructions = []
        self.block_path_instructions = []
        self.block_path_surface_instructions = []
        for block_position in path:
            block = self.blocks[block_position]
            in_vector = block.in_vector
            out_vector = block.out_vector

            in_vector_2D = [in_vector[0], in_vector[2]]
            out_vector_2D = [out_vector[0], out_vector[2]]
            block_instruction = float(np.cross(in_vector_2D, out_vector_2D) * block.block_size)
            block_surface_instruction = traction_for_surface_prefix(block.surface_name)
            self.block_path_instructions.append(block_instruction)
            self.block_path_surface_instructions.append(block_surface_instruction)

            # Keep path instructions tile-aligned with path_tile_index.
            # Some blocks contribute two unique path tiles (for example Curve2/Curve3),
            # so a block-level instruction list drifts out of sync with tile progress.
            for tile in (block.in_tile, block.out_tile):
                if self.path_tiles and np.array_equal(self.path_tiles[-1], tile):
                    continue
                self.path_tiles.append(tile)
                self.path_instructions.append(block_instruction)
                self.path_surface_instructions.append(block_surface_instruction)
        self.path_height_instructions = self.generate_path_height_instructions()
        print("Path tiles constructed...")
        print(self.path_instructions)
        print("Surface instructions:", self.path_surface_instructions)
        print("Height instructions:", self.path_height_instructions)

        print("Path length:", len(self.path_tiles))

    def generate_block_path_transition_index(self):
        self.block_path_index_by_position = {
            tuple(position): index
            for index, position in enumerate(self.block_path)
        }
        self.connected_block_positions = {
            tuple(position): set()
            for position in self.block_path
        }
        for current_position, next_position in zip(self.block_path, self.block_path[1:]):
            current_key = tuple(current_position)
            next_key = tuple(next_position)
            self.connected_block_positions.setdefault(current_key, set()).add(next_key)
            self.connected_block_positions.setdefault(next_key, set()).add(current_key)

    def can_transition_between_blocks(self, from_block: MapBlock | None, to_block: MapBlock | None) -> bool:
        if from_block is None or to_block is None:
            return False
        from_key = tuple(from_block.logical_position)
        to_key = tuple(to_block.logical_position)
        if from_key == to_key:
            return True
        return to_key in self.connected_block_positions.get(from_key, set())

    def generate_path_height_instructions(self) -> list[float]:
        if not self.path_tiles:
            return []
        instructions: list[float] = []
        max_logical_delta = 2.0
        for index, tile in enumerate(self.path_tiles):
            if index >= len(self.path_tiles) - 1:
                instructions.append(0.0)
                continue
            next_tile = self.path_tiles[index + 1]
            delta_y = float(next_tile[1] - tile[1])
            instructions.append(float(np.clip(delta_y / max_logical_delta, -1.0, 1.0)))
        return instructions

    
    def estimated_path_lenght(self):
        return len(self.path_tiles) * MAP_BLOCK_SIZE

    def get_start_position(self):
        return self.blocks[self.start_logical_position].center_point
    
    def get_start_direction(self):
        return self.blocks[self.start_logical_position].out_vector

    def generate_path_mesh(self):

        segments = []
        points = []
        for i in range(1, len(self.path_tiles)):
            tile_from = self.path_tiles[i-1]
            tile_to = self.path_tiles[i]
            point_from = Map.tile_coordinate_to_point(tile_from, dy=2)
            point_to = Map.tile_coordinate_to_point(tile_to, dy=2)
            segments.append([point_from, point_to])

            if i == 1:
                points.append(point_from)
            points.append(point_to)
        self.path_line_mesh = trimesh.load_path(segments)

        points_mesh = trimesh.Trimesh()
        for point in points:
            point_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(point + np.array([0, 0.5, 0])))
            point_mesh.visual.vertex_colors = [0, 255, 255]
            points_mesh += point_mesh
        self.path_points_mesh = points_mesh

    def generate_map_mesh(self):
        # Create a mesh of the map
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_mesh())
        self.mesh = scene.dump(concatenate=True)
        self.mesh.export(f"Maps/Meshes/{self.map_name}.obj")

    def generate_walls_mesh(self):
        # Create a mesh of the walls
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_walls_mesh())
        self.walls_mesh = scene.dump(concatenate=True)

    def generate_sensor_walls_mesh(self):
        # Create the wall mesh used by the vertical-mode lidar.
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_sensor_walls_mesh())
        self.sensor_walls_mesh = scene.dump(concatenate=True)
    
    def generate_road_mesh(self):
        # Create a mesh of the road
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_road_mesh())
        self.road_mesh = scene.dump(concatenate=True) 

    @staticmethod
    def _build_face_neighbors(mesh: trimesh.Trimesh) -> np.ndarray:
        faces = np.asarray(mesh.faces, dtype=np.int32)
        neighbors = -np.ones((len(faces), 3), dtype=np.int32)
        edge_owner: dict[tuple[int, int], tuple[int, int]] = {}

        # Barycentric index i corresponds to the edge opposite vertex i.
        opposite_edges = ((1, 2), (0, 2), (0, 1))
        for face_index, face in enumerate(faces):
            for opposite_vertex_index, (edge_a_idx, edge_b_idx) in enumerate(opposite_edges):
                edge = tuple(sorted((int(face[edge_a_idx]), int(face[edge_b_idx]))))
                owner = edge_owner.get(edge)
                if owner is None:
                    edge_owner[edge] = (face_index, opposite_vertex_index)
                    continue
                other_face_index, other_opposite_vertex_index = owner
                neighbors[face_index, opposite_vertex_index] = int(other_face_index)
                neighbors[other_face_index, other_opposite_vertex_index] = int(face_index)
        return neighbors

    def generate_road_traversal_data(self):
        self.road_traversal_mesh = self.road_mesh.copy()
        # Weld shared seam vertices so face adjacency works across map blocks.
        self.road_traversal_mesh.merge_vertices(digits_vertex=3)
        self.road_traversal_mesh.remove_unreferenced_vertices()
        self.road_face_neighbors = self._build_face_neighbors(self.road_traversal_mesh)


    def get_mesh(self):
        return self.mesh
    
    def get_walls_mesh(self):
        return self.walls_mesh

    def get_sensor_walls_mesh(self):
        return self.sensor_walls_mesh
    
    def get_road_mesh(self):
        return self.road_mesh

    def get_road_traversal_mesh(self):
        return self.road_traversal_mesh

    def get_road_face_neighbors(self):
        return self.road_face_neighbors
    
    def get_path_line_mesh(self):
        return self.path_line_mesh
    
    def get_path_points_mesh(self):
        return self.path_points_mesh

    def plot_map(self):
        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_road_mesh())
            scene.add_geometry(block.get_walls_mesh())
            block.generate_mesh_points()
            scene.add_geometry(block.get_points_mesh())
            #scene.add_geometry(block.in_point_mesh)
            #scene.add_geometry(block.out_point_mesh)        
            #scene.add_geometry(block.in_vector_mesh)
            #scene.add_geometry(block.out_vector_mesh)
        #scene.add_geometry(self.get_path_line_mesh())
        #scene.add_geometry(self.get_path_points_mesh())
           
        scene.show(flags={"cull": False})



if __name__ == "__main__":
    test_map = Map("small_map")
    test_map.plot_map()
