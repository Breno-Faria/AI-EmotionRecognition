import plot
import os
import base64


def decodeAllData(decodeDir):
    sorted_data = plot.get_sorted_data("data.json")
    if sorted_data == None:
        return
    for class_list in sorted_data:
        image_class = class_list[0]["emotion"] 
        image_dir = decodeDir + os.sep + image_class
        plot.emptyFolder(image_dir)
        for i, image in enumerate(class_list):
            image_name = image_dir + os.sep + f'img{i}.png'
            with open(image_name, "wb") as f:
                pngImg = base64.b64decode(image["img"])
                f.write(pngImg)


def main():
    decodeAllData("dataset")


if __name__ == "__main__":
    main()

