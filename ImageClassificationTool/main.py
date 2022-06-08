from PIL import Image
import os

directory = "C:\\Users\hayde\Documents\School\Summer2022\Comp472\Project\Dataset\Medical+mask\Medical mask\Medical Mask\images"
startingIndex = 3600 #CAN BE USED TO START AT A CERTAIN POSITION

outputFolders = ["None","N95","Surgical","Cloth"] #MAKE THESE FOLDERS IN THE IMAGE PATH
counts = [0,0,0,0]

list = os.listdir(directory)

for i in range(startingIndex, len(list)):

    filename = list[i]

    f = os.path.join(directory, filename)
    if os.path.isfile(f):

        im = Image.open(f)
        im.show()

        print("Selection: ")
        selection = int(input())

        if len(counts) <= selection:
            continue
        elif selection < 0:
            break

        folder = os.path.join(directory, outputFolders[selection])
        file = os.path.join(folder, filename)

        counts[selection] += 1

        im.save(file)