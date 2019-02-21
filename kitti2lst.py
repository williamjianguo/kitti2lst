from os import walk
import numpy as np
from gluoncv import utils
import mxnet as mx
from matplotlib import pyplot as plt
def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

file_list = []
#train_path ="label_2"
train_path = ""
val_path = "label_2"
class_name = ["Truck","Van","Cyclist","Car","Misc","Pedestrian","Tram","Person_sitting"]
# all_ids = np.array([0, 1, 2,3,4,5,6,7])
#
# dog_label = [130, 220, 320, 530]
# bike_label = [115, 120, 580, 420]
# car_label = [480, 80, 700, 170]
#
# im_shape = np.array([1242,375,3])
# all_boxes = np.array([dog_label, bike_label, car_label])


for (dirpath, dirnames, filenames) in walk(train_path):
    file_list.extend(filenames)
    break
file_index = 0
is_for_display = 0
with open('val.lst', 'w') as fw:
    # for i in range(4):
    #     line = write_line('dog.jpg', img.shape, all_boxes, all_ids, i)
    #     print(line)
    #     fw.write(line)
    for file in file_list:
        #display calss names
        vis_class_name = []

        file_path = train_path +"/"+file
        with open(file_path, 'r') as f:
            #read line of txt file
            #create img_path
            file_arr = file.split('.')
            img_path = file_arr[0]+'.'+"png"
            #img = mx.image.imread("training/image_2/"+img_path)
            img = mx.image.imread(img_path)
            im_shape = np.array([1242, 375, 3])
            all_box=[]
            all_ids=[]
            #read line of file
            for line in f:
                #print line
                print(line)
                #split line
                line_arr = line.split(' ')
                #add box coordinate
                box =[]
                class_id = []
                #add id value to ids
                str_class = line_arr[0]
                #if class in class_name
                #if class name of line in class_name
                if(str_class in class_name):
                    #all_ids.append(class_name.index(str_class))
                    #return class id
                    class_id = class_name.index(str_class)
                    vis_class_name.append(str_class)
                    #compute minx,miny,maxx,maxy
                    for i in range(4):
                        if(is_for_display):
                            box.append(round(float(line_arr[i + 4])))
                        else:
                            box.append(line_arr[i + 4])
                    #add line array
                    all_box.append(box)
                    all_ids.append(class_id)
        all_box = np.array(all_box)
        all_ids = np.array(all_ids)
        print("all_box",all_box)
        print("all_ids",all_ids)
        print("vis_class_name",vis_class_name)
        #plt.figure(figsize=(16,9))
        #ax1 = plt.subplot(112+i)
        if (is_for_display==1):
            ax = utils.viz.plot_bbox(img, all_box, labels=all_ids, class_names=class_name)

            #plt.rcParams['savefig.dpi'] = 300  
            #plt.rcParams['figure.dpi'] = 300
            '''def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
                          class_names=None, colors=None, ax=None,
                          reverse_rgb=False, absolute_coordinates=True):'''
            #plt.show()
            plt.ion()
            plt.pause(3)
            plt.close()
        all_ids = np.array(all_ids)
        line_str = write_line(img_path, im_shape, all_box, all_ids, file_index)
            #line = write_line('dog.jpg', img.shape, all_boxes, all_ids, i)
        print(line_str)
        fw.write(line_str)
        #print(f.read())
        file_index += 1
