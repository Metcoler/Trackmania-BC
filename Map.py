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

    connection_tiles = {
        "Start": np.array([0, 0, 0]),
        "Finish": np.array([0, 0, 0]),
        "Checkpoint": np.array([0, 0, 0]),
        "End": np.array([0, 0, 0]),
        "Curve": np.array([1, 0, 0]),
    }
    

    def __init__(self, name: str, logical_position: tuple[int], direction: str) -> None:
        print(name, logical_position, direction)
        
        position = np.array([logical_position[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + logical_position[1] * MAP_BLOCK_SIZE // 2, logical_position[2] * MAP_BLOCK_SIZE])
        self.logical_position = np.array(logical_position)
        self.mesh = trimesh.load(f"Meshes/{name}.obj", force="mesh", process=False)
        
        # get block logical size
        self.block_size = 1
        if name[-1].isdigit():
            self.block_size = int(name[-1])
            name = name[:-1]
        name = name.replace("RoadTech", "")

        # color the mesh
    
        if name in MapBlock.block_colors:
            self.set_color(MapBlock.block_colors[name])

        self.color = [np.random.randint(0, 255) for _ in range(3)]
        
        # update position
        angle = self.direction_dictionary[direction]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [self.block_size * 16, 0, self.block_size * 16])
        self.mesh.apply_transform(rotation_matrix)
        self.mesh.apply_translation(position)
        
        # center
        self.center = list(self.mesh.centroid)
        self.center[1] += 5
        self.center = np.array(self.center)
        self.center_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(self.center))
        self.center_mesh.visual.vertex_colors = [255, 0, 0]

        # origin point
        self.origin_mesh = trimesh.creation.box(extents=[1, 1, 1], transform=trimesh.transformations.translation_matrix(position))
        self.origin_mesh.visual.vertex_colors = self.color



    def get_mesh(self):
        return self.mesh
    
    def get_points_mesh(self):
        return self.center_mesh + self.origin_mesh

    def set_color(self, color: list[int]):
        self.mesh.visual.vertex_colors = color
        self.mesh.visual.face_colors = color

        
class Map:
    
    def __init__(self, map_name) -> None:

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
        

        scene = trimesh.Scene()

        for i, block in enumerate(self.blocks.values()):
            scene.add_geometry(block.get_mesh())
            scene.add_geometry(block.get_points_mesh())

        self.mesh = scene.dump(concatenate=True)
        ## TODO export the mesh to a file with materials and colors  
        self.mesh.export(f"Maps/Meshes/{map_name}.obj")




    def get_mesh(self):
        return self.mesh

    def plot_map(self):
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh)
        scene.show()


if __name__ == "__main__":
    test_map = Map("small_map")
    test_map.plot_map()
