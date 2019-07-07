#classes and subclasses to import
import os

#main where the path is set for the directory containing the test images
if __name__ == "__main__":
    mypath = '.\\'
    i=1
    #getting all files in the directory
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".jpg")]
    #iterate over each file in the directory
    for fp in onlyfiles:
        #process the image
        os.rename(fp,'{}.jpg'.format(str(i).zfill(5)))
        i+=1
    print('[INFO] Done Serializing')



