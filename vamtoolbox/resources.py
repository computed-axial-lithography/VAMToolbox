
from argparse import ArgumentError
import pathlib

def load(input:str):
    """
    Loads the filepath to the example images, .stl files, and saved toolbox files installed with the toolbox

    Parameters
    ----------
    input : str
        resource to load, options: "reschart.png", "trifurcatedvasculature.stl", "bear.stl", "thinker.stl", "screwdriver.stl", "ring.stl", "onaxiscylinder.stl", "offaxiscylinder.stl", "seq0imagesdir", "sino0.sino", "video0.mp4",
    
    Returns
    -------
    filepath to resource : str

    """

    options = [
        "reschart.png",
        "trifurcatedvasculature.stl",
        "bear.stl",
        "thinker.stl",
        "screwdriver.stl",
        "ring.stl",
        "onaxiscylinder.stl",
        "offaxiscylinder.stl",
        "seq0imagesdir",
        "sino0.sino",
        "video0.mp4",

    ]

    input = input.lower()

    abs_parent_path = pathlib.Path(__file__).parent

    resources_static_path = abs_parent_path / "./staticresources"


    if input in options:
        resource_path = str(resources_static_path / input)
    else:
        raise ArgumentError(None,"Input resource not recognized. Options include: %s"%(options))
    
    return resource_path
