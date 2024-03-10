import socket
from struct import unpack
import threading
from time import sleep, time

import cv2




def get_data(s: socket.socket):
    data = dict()
    data['speed'] = unpack(b'@f', s.recv(4))[0] # speed
    data['side_speed'] = unpack(b'@f', s.recv(4))[0] # side speed
    data['distance'] = unpack(b'@f', s.recv(4))[0] # distance
    data['x'] = unpack(b'@f', s.recv(4))[0] # x
    data['y'] = unpack(b'@f', s.recv(4))[0] # y
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
    
    return data

# function that captures data from openplanet    
def data_getter_function():
    global data, new_data_recieved
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 9002))
        while True:
            data = get_data(s)
        

if __name__ == "__main__":
    data = {}
    ##new_data_recieved = False
    ##data_getter_thread = threading.Thread(target=data_getter_function, daemon=True)
    ##data_getter_thread.start()


    sleep(0.2) # wait for connection
    
    print("Waiting to recieve some data...")


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 9002))
        start_time = time()
        while True:
            
            data = get_data(s)
            
            """  
            image = camera.grab()
            if image is None:
                continue
            
            image = cv2.resize(image, (854, 480))
            cv2.imshow("Game Capture 1", image)
            key = cv2.waitKey(1)

            if key == ord('q'):  # Press 'q' to quit
                break

            """
            time_passed = time() - start_time
            if time_passed == 0.0:
                time_passed = 0.01
            fps = 1 / time_passed
            start_time = time()  
            ##print(data['x'], data['y'], data['z'])
            print(f"fps: {fps}")


    cv2.destroyAllWindows()    










    