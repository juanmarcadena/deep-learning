# %%

from pycocotools.coco import COCO
import numpy as np
import PIL
import numpy as np
from PIL import Image

np.random.seed(0)

# %%
dataDir='..'
dataType='train2014'
trainAnnFile='annotations/instances_train2014.json'
valAnnFile = 'annotations/instances_val2014.json'

cats = ['airplane', 'bus', 'cat', 'dog', 'pizza']
cocoTrain = COCO(trainAnnFile)
cocoVal = COCO(valAnnFile)

# %%
catIds = cocoTrain.getCatIds(catNms=cats)
imgIdsTrain = [0,0,0,0,0]
imgIdsVal = [0,0,0,0,0]

for i, cat in enumerate(catIds):
    imgIdsTrain[i] = set(cocoTrain.getImgIds(catIds=cat))
    imgIdsVal[i] = set(cocoVal.getImgIds(catIds=cat))
    
imgIdsTrain[0] = imgIdsTrain[0]-(imgIdsTrain[1]|imgIdsTrain[2]|imgIdsTrain[3]|imgIdsTrain[4])
imgIdsTrain[1] = imgIdsTrain[1]-(imgIdsTrain[0]|imgIdsTrain[2]|imgIdsTrain[3]|imgIdsTrain[4])
imgIdsTrain[2] = imgIdsTrain[2]-(imgIdsTrain[0]|imgIdsTrain[1]|imgIdsTrain[3]|imgIdsTrain[4])
imgIdsTrain[3] = imgIdsTrain[3]-(imgIdsTrain[0]|imgIdsTrain[2]|imgIdsTrain[1]|imgIdsTrain[4])
imgIdsTrain[4] = imgIdsTrain[4]-(imgIdsTrain[0]|imgIdsTrain[2]|imgIdsTrain[3]|imgIdsTrain[1])
imgIdsVal[0] = imgIdsVal[0]-(imgIdsVal[1]|imgIdsVal[2]|imgIdsVal[3]|imgIdsVal[4])
imgIdsVal[1] = imgIdsVal[1]-(imgIdsVal[0]|imgIdsVal[2]|imgIdsVal[3]|imgIdsVal[4])
imgIdsVal[2] = imgIdsVal[2]-(imgIdsVal[0]|imgIdsVal[1]|imgIdsVal[3]|imgIdsVal[4])
imgIdsVal[3] = imgIdsVal[3]-(imgIdsVal[0]|imgIdsVal[2]|imgIdsVal[1]|imgIdsVal[4])
imgIdsVal[4] = imgIdsVal[4]-(imgIdsVal[0]|imgIdsVal[2]|imgIdsVal[3]|imgIdsVal[1])


# %%
for i in range(len(cats)):

    trainIds = np.random.choice(list(imgIdsTrain[i]), 1500,replace = False)
    imgsTrain = cocoTrain.loadImgs(trainIds)
    
    for image in imgsTrain:
        # Opens a image in RGB mode
        im = Image.open('train2014.nosync/%s'%(image['file_name']))
        im = im.resize((64, 64))
        im.save('data/train2014.nosync/%s_%s.jpg'%(cats[i],image['id']))
        
    
    valIds = np.random.choice(list(imgIdsVal[i]), 500,replace = False)
    imgsVal = cocoVal.loadImgs(valIds)

    for image in imgsVal:
        im = Image.open('val2014.nosync/%s'%(image['file_name']))
        im = im.resize((64, 64))
        im .save('data/val2014.nosync/%s_%s.jpg'%(cats[i],image['id']))



