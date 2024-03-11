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
    

    def __init__(self, name: str, position: list[float], direction: str) -> None:
        print(name, position, direction)
        self.mesh = trimesh.load(f"Meshes/{name}.obj", force="mesh", process=False)
        self.block_size = 1

        if name[-1].isdigit():
            self.block_size = int(name[-1])

        for block_type, color in self.block_colors.items():
            if block_type in name:
                self.mesh.visual.vertex_colors = color
                self.mesh.visual.face_colors = color
                break
        
        angle = self.direction_dictionary[direction]
        self.mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0], [self.block_size * 16, 0, self.block_size * 16]))
        self.mesh.apply_translation(position)
        

    def get_mesh(self):
        return self.mesh
        
        
class Map:
    def __init__(self, file_name) -> None:
        self.blocks = []
        self.scene = trimesh.Scene()
        self.num_blocks = 0
        
        
        with open(file_name, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '#':
                    continue
                block_name, block_position, block_direction = line.split(";")

                block_position = list(map(int, block_position.split(",")))
                block_position = [block_position[0] * MAP_BLOCK_SIZE, MAP_GROUND_LEVEL + block_position[1] * MAP_BLOCK_SIZE // 2, block_position[2] * MAP_BLOCK_SIZE]
                block = MapBlock(block_name, block_position, block_direction[0])
                self.blocks.append(block)
                self.scene.add_geometry(block.get_mesh())
                self.num_blocks += 1
        self.mesh = self.scene.dump(concatenate=True)

    def callback_function(self, scene: trimesh.Scene):
        pass

    def plot_map(self):
        self.scene.show(callback=self.callback_function)


if __name__ == "__main__":
    test_map = Map("Maps/AI Training.txt")
    test_map.plot_map()
