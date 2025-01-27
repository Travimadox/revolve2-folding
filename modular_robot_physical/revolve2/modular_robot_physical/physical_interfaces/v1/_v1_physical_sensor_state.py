from revolve2.modular_robot.body.base import ActiveHingeSensor
from revolve2.modular_robot.sensor_state import ActiveHingeSensorState

from .._physical_sensor_state import PhysicalSensorState


class V1PhysicalSensorState(PhysicalSensorState):
    """Implements PhysicalSensorState for v1 harware."""

    def get_active_hinge_sensor_state(
        self, sensor: ActiveHingeSensor
    ) -> ActiveHingeSensorState:
        """
        Get sensor states for Hinges.

        :param sensor: The sensor to query.
        :raises NotImplementedError: If there is no implemented sensor of this type.
        """
        raise NotImplementedError("There are no hinge sensors for this hardware.")
