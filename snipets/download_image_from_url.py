import cv2
from PIL import Image
import urllib
import numpy as np


def url_to_image(url, path, name, error_image):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url, timeout=4)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Reading file " + name)
    if image is None:
        raise Exception("Error")
    if image.size == error_image.size and (image == error_image).all():
        raise Exception("Error")
    img_format = Image.fromarray(image, 'RGB')
    img_format.save(path + "/" + name)


def pull_images(filepath, path, label, start, max):
    f = open(filepath, "r", encoding="utf8")
    urls = f.read()
    urls = urls.split("\n")
    # only_urls = [url.split("\t")[1] for url in urls]
    error_image = Image.open("inputs/error_flickr.png")
    error_image = np.array(error_image, dtype=int)
    j = 0

    for i in range(start, len(urls)):
        if j == max:
            return
        try:
            url_to_image(urls[i], path, label + str(i) + ".bmp", error_image)
            j += 1
        except Exception:
            print("Error in file " + label + str(i) + ".bmp")
