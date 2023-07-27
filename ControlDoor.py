import time
import random
def GetDistance():
    # Generate a random distance between 0 and 100 cm
    distance = random.uniform(1, 50)
    print("Random Distance:", distance, "cm")
    return distance


def Detected_Object():
    consecutive_count = 0
    required_consecutive_count = 2
    missed_count = 0
    max_missed_count = 12
    
    while True:
        distance = GetDistance()
        if distance < 30:
            consecutive_count += 1
            missed_count = 0
        else:
            consecutive_count = 0
            missed_count += 1
        
        if consecutive_count >= required_consecutive_count:
            return True
        
        if missed_count >= max_missed_count:
            return False
        
        time.sleep(0.5)

def open_door():
    print("Opening the door")

    time.sleep(2)

    time.sleep(1)
    print("Door is opened")

def close_door():
    print("Closing the door")
    time.sleep(2)

    print("Door is closed")


def on_led():
    print("on led")
def off_led():
    print("on led")
