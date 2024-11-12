import os
from retinaface import Retinaface

retinaface_ec = Retinaface(1)
list_dir = os.listdir("face")
image_paths = []
names = []
for name in list_dir:
    image_paths.append("face/" + name)
    name_without_extension = os.path.splitext(name)[0]
    names.append(name_without_extension)

retinaface_ec.encode_face_dataset(image_paths, names)