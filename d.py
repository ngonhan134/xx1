import RPi.GPIO as GPIO
import time

def GetDistance():
    GPIO.setmode(GPIO.BCM)
    TRIG = 23
    ECHO = 24
    print("Distance Measurement In Progress")
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, False)
    print("Waiting For Sensor To Settle")
    time.sleep(2)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    print("Distance:", distance, "cm")
    GPIO.cleanup()
    return distance

def Detected_Object():
    consecutive_count = 0
    required_consecutive_count = 2
    while True:
        distance = GetDistance()
        if distance < 30:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count >= required_consecutive_count:
            return True
        time.sleep(2)

def open_door():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)
    servo1 = GPIO.PWM(11, 50)
    servo1.start(0)
    print("Opening the door")
    servo1.ChangeDutyCycle(7)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    time.sleep(1)
    print("Door is opened")
    GPIO.cleanup()

def close_door():
    print("Closing the door")
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)
    servo1 = GPIO.PWM(11, 50)
    servo1.start(0)
    servo1.ChangeDutyCycle(2)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    servo1.stop()
    GPIO.cleanup()
    print("Door is closed")

# Test opening and closing the door based on detected object
open_door()
print("Door is open")

while True:
    distance = GetDistance()
    print("Distance:", distance, "cm")
    if distance > 35:
        close_door()
        print("Door is closed")
        break
    time.sleep(0.5)