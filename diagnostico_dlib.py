from pathlib import Path
import numpy as np
import face_recognition
from PIL import Image, ImageOps

root = Path(r"C:\Users\juanm\Sistema_Reconocimiento_Facial\personas")
p = next(root.rglob("*.jpg"), None)
if p is None:
    p = next(root.rglob("*.jpeg"), None)
if p is None:
    raise SystemExit("No se encontro ninguna foto JPG o JPEG")

with Image.open(p) as im:
    rgb = np.array(
        ImageOps.exif_transpose(im).convert("RGB"),
        dtype=np.uint8,
        copy=True,
        order="C"
    )

print("Fotografia:", p.name)
print("NumPy:", np.__version__)
print("Tipo:", rgb.dtype)
print("Dimensiones:", rgb.shape)
print("Memoria compatible:", rgb.flags["C_CONTIGUOUS"])

rostros = face_recognition.face_locations(
    rgb,
    number_of_times_to_upsample=1,
    model="hog"
)
print("Rostros detectados:", len(rostros))
