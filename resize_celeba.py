from PIL import Image

import os



src = "celeba_hq"

dst = "celeba_processed_128"

os.makedirs(dst, exist_ok=True)



for root, dirs, files in os.walk(src):

    for filename in files:

        if filename.endswith(".jpg"):

            src_path = os.path.join(root, filename)

            img = Image.open(src_path)

            img = img.resize((128, 128))

            img.save(os.path.join(dst, filename))



print("Done resizing!")

