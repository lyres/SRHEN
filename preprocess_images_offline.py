import os
import cv2

# Basic setting
IMG_SETS = ['train', 'test', 'valid']
IMG_NUMS = [80000, 40000, 5000]
IMG_SIZE = 320 # new image size
DIR_IMG  = '/data/MSCOCO/'
DIR_OUT  = '/data/MSCOCO_%d/' % IMG_SIZE

for IMG_SET, IMG_NUM in zip(IMG_SETS, IMG_NUMS):
    print('[Resizing %s images]' % IMG_SET)
    
    dir_img = DIR_IMG + IMG_SET + '_images/'
    dir_out = DIR_OUT + IMG_SET + '_images/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    # Get image list
    img_list = os.listdir(dir_img)
    img_list.sort()
    
    # Read, resize and save image
    f_ind = 0
    while f_ind < IMG_NUM:
        img = cv2.imread(dir_img + img_list[f_ind])
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(dir_out + '%06d.jpg' % f_ind, img)
            f_ind += 1
            if not f_ind % 1000:
                print('\tProcess %06d %s images ...' % (f_ind, IMG_SET))
