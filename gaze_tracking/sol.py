from gpiozero import OutputDevice
from time import sleep

RELAY_PIN = 12 
RELAY_POWER_PIN = 2

relay = OutputDevice(RELAY_PIN)
relay_power = OutputDevice(RELAY_POWER_PIN)

relay_power.on()


try:
    while True:
        relay.on()
        sleep(2)  


        relay.off()
        sleep(2)

except KeyboardInterrupt:
    relay.off()
    relay_power.off()