import trimesh
import numpy as np
import trimesh.visual.color
import time

MAP_BLOCK_SIZE = 32
MAP_GROUND_LEVEL = -8.5*(MAP_BLOCK_SIZE // 2)

class MapBlock:
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
        "Curve": (np.array([0, 0, 0]), np.array([1, 0, 1])),
    }

    in_out_directions = {
        "Start": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Finish": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Checkpoint": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Straight": (np.array([0, 0, 1]), np.array([0, 0, 1])),
        "Curve": (np.array([1, 0, 0]), np.array([0, 0, 1])),
    }
    

    def __init__(self, name: str, logical_position: tuple[int], direction: str) -> None:
        
        self.logical_position = np.array(logical_position)
        self.position = np.array([logical_position[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + logical_position[1] * MAP_BLOCK_SIZE // 2, logical_position[2] * MAP_BLOCK_SIZE])
        
        self.mesh = trimesh.load(f"Meshes/{name}.obj", force="mesh", process=False)
        
        # get block logical size
        self.block_size = 1
        if name[-1].isdigit():
            self.block_size = int(name[-1])
            name = name[:-1]
        name = name.replace("RoadTech", "")
        self.name = name 

        self.color = [np.random.randint(0, 255) for _ in range(3)]
        
        # update position
        angle = self.direction_dictionary[direction]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [self.block_size/2 * MAP_BLOCK_SIZE, 0, self.block_size/2 * MAP_BLOCK_SIZE])
        self.mesh.apply_transform(rotation_matrix)
        self.mesh.apply_translation(self.position) 
         
        # center
        self.center_point = list(np.array([16, 0, 16]) + self.position)
        if self.block_size > 1:
            self.center_point = list(self.mesh.centroid)

        self.center_point[1] = self.position[1] + 5
        self.center_point = np.array(self.center_point)
        
        self.find_walls_only()
        self.calculate_in_out_points(name, direction)
        self.calculate_in_out_vectors(name, direction)        
        self.generate_mesh_points()

        # color the mesh
        if name in MapBlock.block_colors:
            self.set_color(MapBlock.block_colors[name])
            

    def calculate_in_out_points(self, name, direction):
        # in and out points
        self.in_title, self.out_title = self.in_out_tiles[name]
        self.in_title = list(self.in_title * (self.block_size-1)) + [1]
        self.out_title = list(self.out_title * (self.block_size-1)) + [1]

        angle = self.direction_dictionary[direction]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [(self.block_size-1)/2, 0, (self.block_size-1)/2])

        self.in_title = np.round(np.array(np.dot(rotation_matrix, self.in_title)[:3])) + self.logical_position
        self.out_title = np.round(np.array(np.dot(rotation_matrix, self.out_title)[:3])) + self.logical_position
    


        self.in_point = np.array([16+self.in_title[0] * MAP_BLOCK_SIZE, self.center_point[1], 16+self.in_title[2] * MAP_BLOCK_SIZE])
        self.out_point = np.array([16+self.out_title[0] * MAP_BLOCK_SIZE, self.center_point[1], 16+self.out_title[2] * MAP_BLOCK_SIZE])

    def calculate_in_out_vectors(self, name, direction):
        in_vector, out_vector = self.in_out_directions[name]
        in_vector = list(in_vector) + [0]
        out_vector = list(out_vector) + [0]

        angle = self.direction_dictionary[direction]

        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [0, 0, 0])
        self.in_vector = np.round(np.dot(rotation_matrix, in_vector)[:3])
        self.out_vector = np.round(np.dot(rotation_matrix, out_vector)[:3])


    def swap_in_out(self):
        self.in_title, self.out_title = self.out_title, self.in_title
        self.in_point, self.out_point = self.out_point, self.in_point
        self.in_vector, self.out_vector = -self.out_vector, -self.in_vector
        self.generate_mesh_points()
    
    def generate_mesh_points(self):
        ## center
        self.center_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(self.center_point))
        self.center_mesh.visual.vertex_colors = [255, 0, 0]

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

    def find_walls_only(self):
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


        # Create a new mesh with faces not pointing in the y-axis direction
        self.walls_mesh = trimesh.Trimesh(vertices=self.mesh.vertices,
                                    faces=self.mesh.faces[walls_indices])
        self.walls_mesh.visual.vertex_colors = [50, 50, 50]
        self.road_mesh = trimesh.Trimesh(vertices=self.mesh.vertices,
                                    faces=self.mesh.faces[road_indices])


    def get_mesh(self):
        return self.mesh
    
    def get_road_mesh(self):
        return self.road_mesh
    
    def get_walls_mesh(self):
        return self.walls_mesh
    
    def get_points_mesh(self):
        return self.in_point_mesh + self.out_point_mesh + self.in_vector_mesh


    def set_color(self, color: list[int]):
        self.mesh.visual.vertex_colors = color
        self.mesh.visual.face_colors = color
        self.road_mesh.visual.vertex_colors = color

    
    def get_out_position_vector(self, input_logical_position, input_vector, i=0):
        if i > 2:
            raise Exception("Error: path is not continuous")
        if all(np.array(input_logical_position == self.in_title)) and all(input_vector == self.in_vector):
            return self.out_title, self.out_vector
        self.swap_in_out()
        return self.get_out_position_vector(input_logical_position, input_vector, i+1)
    
    def contains_in_out_title(self, logical_position):
        return all(logical_position == self.in_title) or all(logical_position == self.out_title)
    
    def __repr__(self) -> str:
        return f"{self.name}{self.block_size if self.block_size > 1 else ''}: {self.logical_position}"

        
class Map:
    
    def __init__(self, map_name) -> None:
        print("Creating map...")
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
                elif "End" in logical_position:
                    self.end_logical_position = logical_position
        
        self.construct_path()

        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_mesh())
        self.mesh = scene.dump(concatenate=True)
        self.mesh.export(f"Maps/Meshes/{map_name}.obj")

        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_walls_mesh())
        self.walls_mesh = scene.dump(concatenate=True)

        scene = trimesh.Scene()
        for block in self.blocks.values():
            scene.add_geometry(block.get_road_mesh())
        self.road_mesh = scene.dump(concatenate=True)  



        
    def construct_path(self):
        path = [self.start_logical_position]
        
        logical_position = tuple(self.start_logical_position)
        in_position = self.start_logical_position
        in_vector = self.blocks[logical_position].out_vector
        

        while self.blocks[logical_position].name != "Finish":

            out_position, out_vector = self.blocks[logical_position].get_out_position_vector(in_position, in_vector)
            next_in_position = out_position + out_vector
            
            for block in self.blocks.values():
                if block.contains_in_out_title(next_in_position):
                
                    in_position = next_in_position
                    in_vector = out_vector
                    logical_position = tuple(block.logical_position)
                    path.append(logical_position)
                    break
            else:
                print("Error: path is not continuous")
                break
        print("Path constructed...")
        self.block_path = path
        path_points_tmp = []
        for block_position in path:
            path_points_tmp.append(self.blocks[block_position].in_title)
            path_points_tmp.append(self.blocks[block_position].center_point)
            path_points_tmp.append(self.blocks[block_position].out_title)
        
        self.path_points = []
        for point in path_points_tmp:
            if self.path_points and all(self.path_points[-1] == point):
                continue
            self.path_points.append(point)
        
    def get_start_position(self):
        return self.blocks[self.start_logical_position].center_point
    
    def get_start_direction(self):
        return self.blocks[self.start_logical_position].out_vector

    def generate_path_mesh(self):
        segments = []
        for i in range(1, len(self.path_points)):
            point_from = self.path_points[i-1]
            point_to = self.path_points[i]
            segments.append([point_from, point_to])
        path_mesh = trimesh.load_path(segments)

        return path_mesh



    def get_mesh(self):
        return self.mesh
    
    def get_walls_mesh(self):
        return self.walls_mesh
    
    def get_road_mesh(self):
        return self.road_mesh

    def plot_map(self):
        scene = trimesh.Scene()
        for block in self.blocks.values():
            pass
            scene.add_geometry(block.get_road_mesh())
            scene.add_geometry(block.get_walls_mesh())
            #block.generate_mesh_points()
            #scene.add_geometry(block.get_mesh())
            #scene.add_geometry(block.center_mesh)
            #scene.add_geometry(block.in_point_mesh)
            #scene.add_geometry(block.out_point_mesh)        
            #scene.add_geometry(block.in_vector_mesh)
            #scene.add_geometry(block.out_vector_mesh)
        #scene.add_geometry(self.generate_path_mesh())
           
        scene.show()



if __name__ == "__main__":
    test_map = Map("AI Training #3")
    test_map.plot_map()
