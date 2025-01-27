"""Main script for the example."""

from revolve2.ci_group import terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator


def make_body() -> (
    tuple[BodyV1, ActiveHingeV1, ActiveHingeV1, ActiveHingeV1, ActiveHingeV1]
):
    """
    Create a body for the robot.

    :returns: The created body and references to each hinge: first_left_active_hinge, second_left_active_hinge, first_right_active_hinge, second_right_active_hinge.
    """
    body = BodyV1()
    first_left_active_hinge = ActiveHingeV1(RightAngles.DEG_0)
    second_left_active_hinge = ActiveHingeV1(RightAngles.DEG_0)
    first_right_active_hinge = ActiveHingeV1(RightAngles.DEG_0)
    second_right_active_hinge = ActiveHingeV1(RightAngles.DEG_0)
    body.core.left = first_left_active_hinge
    first_left_active_hinge.attachment = second_left_active_hinge
    second_left_active_hinge.attachment = BrickV1(RightAngles.DEG_0)
    body.core.right = first_right_active_hinge
    first_right_active_hinge.attachment = second_right_active_hinge
    second_right_active_hinge.attachment = BrickV1(RightAngles.DEG_0)
    return (
        body,
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
    )


class CustomBrainInstance(BrainInstance):
    """
    The actual object that controls the robot.

    Created by the `CustomBrain` class.
    """

    first_left_active_hinge: ActiveHingeV1
    second_left_active_hinge: ActiveHingeV1
    first_right_active_hinge: ActiveHingeV1
    second_right_active_hinge: ActiveHingeV1

    def __init__(
        self,
        first_left_active_hinge: ActiveHingeV1,
        second_left_active_hinge: ActiveHingeV1,
        first_right_active_hinge: ActiveHingeV1,
        second_right_active_hinge: ActiveHingeV1,
    ) -> None:
        """
        Initialize the Object.

        :param first_left_active_hinge: First left active Hinge.
        :param second_left_active_hinge: Second left active Hinge.
        :param first_right_active_hinge: First right active Hinge.
        :param second_right_active_hinge: Second right active Hinge.
        """
        self.first_left_active_hinge = first_left_active_hinge
        self.second_left_active_hinge = second_left_active_hinge
        self.first_right_active_hinge = first_right_active_hinge
        self.second_right_active_hinge = second_right_active_hinge

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the modular robot.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """
        control_interface.set_active_hinge_target(self.first_left_active_hinge, 1.0)
        control_interface.set_active_hinge_target(self.second_left_active_hinge, 0.0)
        control_interface.set_active_hinge_target(self.first_right_active_hinge, 0.0)
        control_interface.set_active_hinge_target(self.second_right_active_hinge, -1.0)


class CustomBrain(Brain):
    """
    This is our custom brain.

    It stores references to each hinge of the robot body so they can be controlled individually.
    A brain has a function `make_instance`, which creates the actual object that controls a robot.
    """

    first_left_active_hinge: ActiveHingeV1
    second_left_active_hinge: ActiveHingeV1
    first_right_active_hinge: ActiveHingeV1
    second_right_active_hinge: ActiveHingeV1

    def __init__(
        self,
        first_left_active_hinge: ActiveHingeV1,
        second_left_active_hinge: ActiveHingeV1,
        first_right_active_hinge: ActiveHingeV1,
        second_right_active_hinge: ActiveHingeV1,
    ) -> None:
        """
        Initialize the Object.

        :param first_left_active_hinge: First left active Hinge.
        :param second_left_active_hinge: Second left active Hinge.
        :param first_right_active_hinge: First right active Hinge.
        :param second_right_active_hinge: Second right active Hinge.
        """
        self.first_left_active_hinge = first_left_active_hinge
        self.second_left_active_hinge = second_left_active_hinge
        self.first_right_active_hinge = first_right_active_hinge
        self.second_right_active_hinge = second_right_active_hinge

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return CustomBrainInstance(
            self.first_left_active_hinge,
            self.second_left_active_hinge,
            self.first_right_active_hinge,
            self.second_right_active_hinge,
        )


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Create a body for the robot.
    (
        body,
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
    ) = make_body()

    # Create a brain for the robot.
    brain = CustomBrain(
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
    )

    # Combine the body and brain into a modular robot.
    robot = ModularRobot(body, brain)

    # Create the scene.
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    # Simulate the scene.
    simulator = LocalSimulator()
    simulate_scenes(
        simulator=simulator,
        batch_parameters=make_standard_batch_parameters(),
        scenes=scene,
    )


if __name__ == "__main__":
    main()
