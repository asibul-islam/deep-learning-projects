from PIL import Image
import os

folders = [
    "../data/cats_dogs/train/cats",
    "../data/cats_dogs/train/dogs",
    "../data/cats_dogs/val/cats",
    "../data/cats_dogs/val/dogs"
]

removed = 0
converted = 0

for folder in folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print("Removing non-image:", path)
            os.remove(path)
            removed += 1
            continue

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.save(path, "JPEG")
                converted += 1

        except Exception as e:
            print("Removing corrupted file:", path, e)
            os.remove(path)
            removed += 1

print("Done.")
print("Converted to RGB:", converted)
print("Removed files:", removed)