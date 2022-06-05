from PIL import Image
import os
import cv2
import torchvision.transforms as transforms

size = 100
def square(old_im):
    old_size = old_im.size
    big = old_size[0]if old_size[0]>old_size[1] else old_size[1]
    new_size = (big, big)
    new_im = Image.new("RGB", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
                      (new_size[1]-old_size[1])//2))
    return new_im
#https://stackoverflow.com/questions/14461905/python-if-else-short-hand
def dimensionalNormalize(old_im):
    old_size = old_im.size
    new_size = (size, size)

    new_im = old_im.resize(new_size)
    return new_im
#https://www.youtube.com/watch?v=cWHW9MnX_F4
def normalize(old_im):
    new_im=square(old_im)
    new_im=dimensionalNormalize(new_im)
    return new_im

def GetPhotos(directory):
    dirName="photos"
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    list = os.listdir(directory)

    for i in range(0, len(list)):

        filename = list[i]
        f = os.path.join(directory, filename)     
        
        if os.path.isfile(f):
            im = Image.open(f)
            new_im = normalize(im)
            folder = os.path.join(directory, dirName)
            file = os.path.join(folder, filename)
            new_im.save(file)        


def renameNoMask():

    folder = 'noMask'
    directory = "C:\\Users\hayde\Documents\Test"


    outputFolders = ["None","N95","Surgical","Cloth"] #MAKE THESE FOLDERS IN THE IMAGE PATH
    counts = [0,0,0,0]

    list = os.listdir(directory)

    for i in range(0, len(list)):

        filename = list[i]

        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            os.rename(f, (folder+str(i)+'.jpg'))
            
if __name__ == "__main__":
    #for i in range(len(list)):
        
    #    filename = list[i]
    #   f = os.path.join(directory, filename)
    #    if os.path.isfile(f):

    GetPhotos("C:\\Users\hayde\Documents\Test")

