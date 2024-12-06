# import the other required libraries
import numpy as np
import os
from imgbeddings import imgbeddings
from PIL import Image

# load `imgbeddings` so we can calculate embeddings
ibed = imgbeddings()

for filename in os.listdir("stored-faces"):
    # read the image
    img = Image.open("stored-faces/" + filename)
    # calculate the embedding for this face
    embedding = ibed.to_embeddings(img)[0]
    # and write it to PostgreSQL