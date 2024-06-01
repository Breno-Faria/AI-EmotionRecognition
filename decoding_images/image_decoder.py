import base64
import os

def image_decoder(img_json_array):
    
    emptyFolder("./output_images")
 
    for i, image in enumerate(img_json_array):
        with open(f"./output_images/img{i}.png", "wb") as fo:
            pngImg = base64.b64decode(image)
            fo.write(pngImg)

def emptyFolder(path):
    for img in os.listdir(path):
        os.remove(path+"/"+img)
