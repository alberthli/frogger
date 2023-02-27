# noqa
import os

import matplotlib.pyplot as plt
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder


def model_inspector(filename):
    """Basic model inspector for verifying network permissions."""
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.AddModelFromFile(filename)
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"),
    )
    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    diagram = builder.Build()
    sliders.Run(diagram)


iiwa7_model_file = FindResourceOrThrow(
    "drake/manipulation/models/" "iiwa_description/iiwa7/iiwa7_with_box_collision.sdf"
)
model_inspector(iiwa7_model_file)

# verifying display permissions
plt.figure()
plt.title("Test Plot: Success!")
plt.show()

# verifying shared read/write permissions
if os.path.isfile("test.png"):
    os.remove("test.png")
    print("Test image removed.")
else:
    plt.savefig("test.png")
    print(
        "Test image successfully created! Manually delete to ensure shared read/write permissions have been set correctly. If you cannot remove it, running this script again will remove it."
    )
