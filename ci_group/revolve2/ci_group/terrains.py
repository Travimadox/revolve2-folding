"""Standard terrains."""

import math

import numpy as np
import numpy.typing as npt
from noise import pnoise2
from pyrr import Quaternion, Vector3,Matrix44
from revolve2.simulation import Terrain
from revolve2.simulation.running import geometry


def flat(size: Vector3 = Vector3([20.0, 20.0, 0.0])) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            geometry.Plane(
                position=Vector3(),
                orientation=Quaternion(),
                size=size,
            )
        ]
    )


def crater(
    size: tuple[float, float],
    ruggedness: float,
    curviness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    r"""
    Create a crater-like terrain with rugged floor using a heightmap.

    It will look like::

        |            |
         \_        .'
           '.,^_..'

    A combination of the rugged and bowl heightmaps.

    :param size: Size of the crater.
    :param ruggedness: How coarse the ground is.
    :param curviness: Height of the edges of the crater.
    :param granularity_multiplier: Multiplier for how many edges are used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heightmap(num_edges=num_edges)

    max_height = ruggedness + curviness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged + curviness * bowl) / (ruggedness + curviness)

    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: tuple[float, float],
    num_edges: tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    """
    Create a rugged terrain heightmap.

    It will look like::

        ..^.__,^._.-.

    Be aware: the maximum height of the heightmap is not actually 1.
    It is around [-1,1] but not exactly.

    :param size: Size of the heightmap.
    :param num_edges: How many edges to use for the heightmap.
    :param density: How coarse the ruggedness is.
    :returns: The created heightmap as a 2 dimensional array.
    """
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def bowl_heightmap(
    num_edges: tuple[int, int],
) -> npt.NDArray[np.float_]:
    r"""
    Create a terrain heightmap in the shape of a bowl.

    It will look like::

        |         |
         \       /
          '.___.'

    The height of the edges of the bowl is 1.0 and the center is 0.0.

    :param num_edges: How many edges to use for the heightmap.
    :returns: The created heightmap as a 2 dimensional array.
    """
    return np.fromfunction(
        np.vectorize(
            lambda y, x: (x / num_edges[0] * 2.0 - 1.0) ** 2
            + (y / num_edges[1] * 2.0 - 1.0) ** 2
            if math.sqrt(
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
            )
            <= 1.0
            else 0.0,
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )

def slope(size: tuple[float, float], angle_deg: float = 15.0) -> Terrain:
    """
    Create a slope terrain with a flat area at the end.

    :param size: Size of the slope (width, length)
    :param angle_deg: The angle of the slope in degrees.
    :returns: The created terrain.
    """

    # Calculate rotation quaternion based on the slope angle
    rotation_matrix = Matrix44.from_eulers((0.0, np.radians(angle_deg), 0.0))
    rotation_quat = Quaternion.from_matrix(rotation_matrix)

    # Create the slope terrain
    return Terrain(
        static_geometry=[
            geometry.Plane(
                position=Vector3([0.0, 0.0, 0.0]),  # Start position
                orientation=rotation_quat,          # Orientation based on the slope
                size=Vector3([size[0], size[1], 0.0]),  # Size
            )
            # You can add a flat terrain at the end here
        ]
    )

def slope_with_flat(size_slope: tuple[float, float], size_flat: tuple[float, float], angle_deg: float = 15.0) -> Terrain:
    """
    Create a slope terrain with a flat area at the end.

    :param size_slope: Size of the slope (width, length)
    :param size_flat: Size of the flat terrain (width, length)
    :param angle_deg: The angle of the slope in degrees.
    :returns: The created terrain.
    """
    rotation_matrix = Matrix44.from_eulers((0.0, np.radians(angle_deg), 0.0))
    rotation_quat = Quaternion.from_matrix(rotation_matrix)
    
    # Assuming the slope starts at the origin, calculate the end position of the slope.
    end_slope_y = size_slope[1] * np.cos(np.radians(angle_deg))
    end_slope_z = -size_slope[1] * np.sin(np.radians(angle_deg))
    
    # Position of the flat terrain should start where the slope ends
    flat_position = Vector3([0.0, end_slope_y, end_slope_z])

    return Terrain(
        static_geometry=[
            geometry.Plane(
                position=Vector3([0.0, 0.0, 0.0]),  # Slope starts at origin
                orientation=rotation_quat,          # Orientation based on the slope
                size=Vector3([size_slope[0], size_slope[1], 0.0]),  # Size
            ),
            geometry.Plane(
                position=flat_position,  # Flat terrain starts where slope ends
                orientation=Quaternion(), # No rotation
                size=Vector3([size_flat[0], size_flat[1], 0.0]),  # Size
            ),
        ]
    )

