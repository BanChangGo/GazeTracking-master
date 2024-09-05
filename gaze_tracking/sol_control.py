from gpiozero import OutputDevice
from time import sleep

class RelayController:
    def __init__(self, relay_pin, relay_power_pin):
        self.relay = OutputDevice(relay_pin)
        self.relay_power = OutputDevice(relay_power_pin)
        self.relay_power.on()  # Turn on the power for the relay

    def on(self):
        self.relay.on()

    def off(self):
        self.relay.off()

    def cleanup(self):
        self.relay.off()
        self.relay_power.off()

# Example usage:
if __name__ == "__main__":
    relay_controller = RelayController(relay_pin=12, relay_power_pin=2)

    '''
    try:
        while True:
            relay_controller.on()
            sleep(2)
            relay_controller.off()
            sleep(2)

    except KeyboardInterrupt:
        relay_controller.cleanup()
    '''