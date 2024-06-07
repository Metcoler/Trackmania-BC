from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import trimesh
import threading

from Enviroment import RacingGameEnviroment

def callback_function(scene: trimesh.Scene):
    # Callback function is called every frame of visualization
    enviroment.car.visualize_rays(scene)
    enviroment.car.update_model_view()
    enviroment.car.update_camera(scene)


def plot_map():
    # Create a scene with the car and the map
    scene = trimesh.Scene()
    scene.add_geometry(enviroment.car.get_mesh())
    scene.add_geometry(enviroment.map.get_walls_mesh())
    scene.add_geometry(enviroment.map.get_road_mesh())
    scene.add_geometry(enviroment.map.get_path_line_mesh())
    scene.add_geometry(enviroment.map.get_path_poins_mesh())
    scene.show(callback=callback_function)


trained_model = "third_stage_training"
map_name = "small_map"
model = PPO.load(f"logs/{trained_model}.zip")
vizualize = False

enviroment = RacingGameEnviroment(map_name=map_name, never_quit=True)

observation, info = enviroment.reset()
done = False


if vizualize:
    window_thread = threading.Thread(target=plot_map, daemon=True)
    window_thread.start()

while not done:
    action, _ = model.predict(observation, deterministic=False)
    observation, reward, done, truncated, info = enviroment.step(action)
enviroment.close()
