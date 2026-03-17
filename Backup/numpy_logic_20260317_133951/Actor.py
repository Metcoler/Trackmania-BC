import inputs
import time
import threading
import socket
import keyboard

from Car import Car
from Map import Map

class PlayerController:
    def __init__(self):
        self.throttle = 0
        self.brake = 0
        self.steering = 0

        self.alive = True

        self.thread = threading.Thread(target=self.thread_function, daemon=True)
        self.thread.start()

    
    def update(self):
        try:
            events = inputs.get_gamepad()
        except Exception as e:
            self.alive = False
            return
    
        for event in events:
            if event.ev_type == 'Absolute':
                if event.code == 'ABS_RZ':
                    self.throttle = event.state / 255
                    self.throttle = round(self.throttle, 2)
                    

                elif event.code == 'ABS_Z':
                    self.brake = event.state / 255
                    self.brake = round(self.brake, 2)
    
                elif event.code == 'ABS_X':
                    self.steering = event.state / 32768
                    self.steering = self.steering if abs(self.steering) > 0.1 else 0
                    self.steering = round(self.steering, 2)
    def thread_function(self):
        while self.alive:
            self.update()
    
    def get_values(self):
        return [self.throttle, self.brake, self.steering]

    def __repr__(self) -> str:
        return f"Throttle: {self.throttle}, Brake: {self.brake}, Steering: {self.steering}"



def print_fps(frame: int):
    # Print the average FPS every 100 frames
    global start_time_fps
    if frame % 100 != 0:
        return
    
    dt = time.time() - start_time_fps
    if dt == 0:
        dt = 0.01
    print(f"FPS: {(100/dt):02f}", end="\r")
    start_time_fps = time.time()
    
# Main function
if __name__ == "__main__":
    map_name = "AI Training #3"
    controller = PlayerController()
    game_map = Map(map_name)
    car = Car(game_map)
    start_time_fps = time.time()

    while True:
        print("===============================================")
        print("Press 'x' to exit space to start data gathering process...")
        key = keyboard.read_event()
        if key.name == "x":
            print("Exiting...")
            break
        
        elif key.name != "space":
            continue
        
        print("Starting data gathering process...")

        
        # start data gathering process
        data_buffer = []
        frame = 0
        while True: 
            if keyboard.is_pressed("x"):
                print("Exiting data gathering process...")
                break
            

            distances, data_dictionary = car.get_data()
            data_buffer.append((distances, data_dictionary, controller.get_values()))
            print_fps(frame)
            frame += 1
            if data_dictionary["done"] == 1.0:
                break
        time.sleep(1)
        print("Do you want to save the data? (y/n)")
        
        if keyboard.read_event().name == "y":
            print(f"Saving {len(data_buffer)} data samples...")
            with open("driver_data.txt", "a") as file:
                ## write first line
                distances, data_dictionary, player_actions = data_buffer[0]
                print(len(distances), *data_dictionary.keys(), "throttle", "brake", "steering", sep=",", file=file)


                for distances, data_dictionary, player_actions in data_buffer:
                    # write distances
                    print(*distances, sep=",", end=";",file=file)
                    # write data dictionary
                    print(*data_dictionary.values(), sep=",", end=";", file=file)
                    # write player actions
                    print(*player_actions, sep=",",file=file)
                    
            print("Data saved")
        else:
            print("Data discarded...")

            
        
