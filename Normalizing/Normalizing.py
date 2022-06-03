from PIL import Image
import os
def square(old_im):
    old_size = old_im.size
    big = old_size[0]if old_size[0]>old_size[1] else old_size[1]
    new_size = (big, big)
    new_im = Image.new("RGB", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
                      (new_size[1]-old_size[1])//2))

    new_im.show()
    new_im.save('someimage.jpg')
#https://stackoverflow.com/questions/14461905/python-if-else-short-hand
def dimensionalNormalize(old_im):
    old_size = old_im.size
    
#https://www.youtube.com/watch?v=cWHW9MnX_F4
def normalize(old_im):
    square(old_im)
    dimensionalNormalize(old_im)
    
if __name__ == "__main__":
    directory = "C:\\Users\hayde\Documents\School\Summer2022\Comp472\Project\test"
    
    list = os.listdir(directory)
    for i in range(len(list)):
        
        filename = list[i]
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            normalize(f)   