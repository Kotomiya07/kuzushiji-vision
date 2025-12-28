from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("kuzushiji").project("column-wnbs7")
version = project.version(8)
dataset = version.download("yolov12")
