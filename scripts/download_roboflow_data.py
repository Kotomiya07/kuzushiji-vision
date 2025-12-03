from roboflow import Roboflow

rf = Roboflow(api_key="Tp3RvoZak6Po5XotNvK0")
project = rf.workspace("kuzushiji").project("column-wnbs7")
version = project.version(8)
dataset = version.download("yolov12")
