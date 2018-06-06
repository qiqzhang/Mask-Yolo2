from base.base_data_generator import *
import os

def getdir(img_dir, suffix, fulldirlist):
    dir_list = os.listdir(img_dir)  # 该文件夹下所有的文件（包括文件夹）
    for eachDir in dir_list:  # 遍历当前文件夹内所有文件
        full_dir = os.path.join(img_dir, eachDir)
        if os.path.isfile(full_dir):
            # 只读取给定后缀名的文件名
            if eachDir.endswith('.' + suffix):
                fulldirlist.append(full_dir)  # [(dir,stand, 1), (dir, stand, 10), ...]
        elif os.path.isdir(full_dir):
            getdir(full_dir, suffix, fulldirlist)
    return fulldirlist


class DataGenerator(BaseDataGenerator):

    def get_images_labels_list(self):
        images_list=[]
        labels_list=[]
        with open(self.config.train_txt,'r') as f:
            for line in f:
                image_full_name=line.split('\n')[0]
             #   print(image_full_name)
                images_list.append(image_full_name)
                label_name = line.split('/')[-1].split('.')[0]+'.txt'
                label_full_name=os.path.join(self.config.labels_dir,label_name)
             #   print(label_full_name)
                labels_list.append(label_full_name)
        dataset_size = len(images_list)
        print(len(images_list),len(labels_list),dataset_size)
        return images_list,labels_list,dataset_size

    def get_labels_list(self):
        with open(self.config.labels_dir, 'r') as f:
            labels = f.read().split('\n')
        labels = [int(label) for label in labels]
        return labels

    def parse_function(self,filename, label):
        image_string = tf.read_file(filename)
        label_b = tf.read_file(label)
        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize_images(image, [self.config.net_size_w, self.config.net_size_h])
        return image, label_b
