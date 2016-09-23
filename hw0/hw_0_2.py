from PIL import Image
import sys

image = Image.open(sys.argv[1])
image.rotate(180).save("ans2.png")
