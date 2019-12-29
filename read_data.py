import cv2, os, glob, re, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

start = time.time()
def splitstring(word):
    label = np.zeros(15)
    x = re.split(" ", word)
    if(x[0] != 'nonface'):
        personId = int(x[1][6:8])
        label[personId-1] = 1
        # if(personId == 1):
        #     label[0] = 1
        # elif(personId == 4):
        #     label[1] = 1
        # else:
        #     label[2] = 1
    return label

# read several image
img_dir = "15-people" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*jpg')
files = glob.glob(data_path)

img_dir = "15-people" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*png')
files_png = glob.glob(data_path)

data = []
label = []

for f1 in files:
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    title = splitstring(base[0])
    # print(title)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(384,288))
    data.append(image)
    label.append(title)

for f1 in files_png:
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    title = splitstring(base[0])
    # print(title)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(384,288))
    data.append(image)
    label.append(title)

data = np.array(data)
label = np.array(label)
print(data.shape, label.shape)
end = time.time()
print("read data complete " , round(end-start,2) , "s, total : ", len(data))