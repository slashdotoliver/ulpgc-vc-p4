# %% [markdown]
#
# un detector de objetos, que permita localizar vehículos y personas
# un localizador de matrículas
#


# %%

import cv2  
import math 

from ultralytics import YOLO


model = YOLO('yolo11n-seg.pt')


classNames = ["person", "bicycle", "car", "bus"]

filename = "C0142.mp4"
results = model.track(source=filename, show=True)
# %%
