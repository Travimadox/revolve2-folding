import math

import pigpio

from revolve2.modular_robot.body.base import ActiveHinge

from ..._uuid_key import UUIDKey
from .._physical_control_interface import PhysicalControlInterface, Pin


class V1PhysicalControlInterface(PhysicalControlInterface):
    """Implements PhysicalControlInterface for v1 hardware."""

    _PWM_FREQUENCY = 50
    _gpio: pigpio.pi

    _CENTER = 157.0
    _ANGLE60 = 64.0

    def __init__(
        self,
        debug: bool,
        dry: bool,
        hinge_mapping: dict[UUIDKey[ActiveHinge], int],
        inverse_pin: dict[int, bool],
    ) -> None:
        """
        Initialize the PhysicalInterface.

        :param debug: If debugging messages are activated.
        :param dry: If dry.
        :param hinge_mapping: The modular robots hinges mapped to servos of the physical robot.
        :param inverse_pin: If pins are inversed.
        :raises RuntimeError: If GPIOs could not initialize.
        """
        super().__init__(
            dry=dry, debug=debug, hinge_mapping=hinge_mapping, inverse_pin=inverse_pin
        )

        if not self._dry:
            self._gpio = pigpio.pi()
            if not self._gpio.connected:
                raise RuntimeError("Failed to reach pigpio daemon.")

        self._pins = [
            Pin(pin_id, self._inverse_pin.get(pin_id, False))
            for pin_id in hinge_mapping.values()
        ]

        if self._debug:
            print(f"Using PWM frequency {self._PWM_FREQUENCY}Hz")

        if not self._dry:
            try:
                for pin in self._pins:
                    self._gpio.set_PWM_frequency(pin.pin, self._PWM_FREQUENCY)
                    self._gpio.set_PWM_range(pin.pin, 2048)
                    self._gpio.set_PWM_dutycycle(pin.pin, 0)
            except AttributeError as err:
                raise RuntimeError("Could not initialize gpios.") from err

    def stop_pwm(self) -> None:
        """Stop the signals and the robot."""
        if self._debug:
            print(
                "Turning off all pwm signals for pins that were used by this controller."
            )
        for pin in self._pins:
            if not self._dry:
                self._gpio.set_PWM_dutycycle(pin.pin, 0)

    def _set_servo_target(self, pin: Pin, target: float) -> None:
        """
        Set the target for a single Servo.

        :param pin: The servos pin.
        :param target: The target angle.
        """
        if self._debug:
            print(f"{pin.pin:03d} | {target}")

        if not self._dry:
            invert_mul = 1.0 if pin.invert else -1.0

            angle = (
                self._CENTER
                + invert_mul * target / (1.0 / 3.0 * math.pi) * self._ANGLE60
            )
            self._gpio.set_PWM_dutycycle(pin.pin, angle)
