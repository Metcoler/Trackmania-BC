import trimesh
import numpy as np
MAP_BLOCK_SIZE = 32

class MapBlock:
    direction_dictionary = {
        "N": 180,
        "E": 90,
        "S": 0,
        "W": 270
    }
    def __init__(self, name, position, direction) -> None:
        print(name, position, direction)
        self.mesh = trimesh.load(f"Meshes/{name}.obj", force="mesh")

        angle = self.direction_dictionary[direction]
        if "Curve" in name:
            angle -= 90

        self.mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [16, 0, -16]))
        self.mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0], [16, 0, -16]))
        self.mesh.apply_translation(position)
        
        

    def get_mesh(self):
        return self.mesh
        





class Map:
    def __init__(self, file_name) -> None:
        self.blocks = []
        self.num_blocks = 0

        with open(file_name, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '#':
                    continue
                block_name, block_position, block_direction = line.split(";")
                block_position = list(map(lambda coord: int(coord) * MAP_BLOCK_SIZE, block_position.split(",")))
                self.blocks.append(MapBlock(block_name, block_position, block_direction[0]))
                self.num_blocks += 1


    def plot_map(self):
        scene = trimesh.Scene()  
        
        for block in self.blocks:
            scene.add_geometry(block.get_mesh())
        
        scene.show()
        

if __name__ == "__main__":
    test_map = Map("Maps/small_map_test_2.txt")
    test_map.plot_map()