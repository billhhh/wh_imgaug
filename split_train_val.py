import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle

list_path = './'
train_list = 'train.txt'
val_list = 'val.txt'
val_ratio = 0.8

src_dir = './newFood_724_clean'

class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names

lst.sort()
print(lst)

with open(os.path.join(list_path,'food_id.txt'), 'w') as f:
    for ind in range(0,len(lst)):
        f.writelines(str(ind) + ":" + lst[ind] + '\n')

f_train = open(os.path.join(list_path, train_list), 'w')
f_val = open(os.path.join(list_path, val_list), 'w')

for ind in range(0,len(lst)):
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    shuffle(sblst)
    print(len(sblst))
    cnt=0
    for pic_name in sblst:
        filepath_src= src_dir + '/' + lst[ind] + '/' + pic_name
        if pic_name.endswith('.db') == True:
            continue
        else:
            cnt+=1
            #if cnt <= 150:
            if cnt <= floor(len(sblst) * val_ratio) and cnt <= 500:
                f_train.writelines(filepath_src + " " + str(ind) + '\n')
            else:
                f_val.writelines(filepath_src + " " + str(ind) + '\n')

f_train.close()
f_val.close()
