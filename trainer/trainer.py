from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import math




class Trainer(BaseTrain):

    def iou_wh(self,r1, r2):
        #print(r1,r2)
        min_w = min(r1[0], r2[0])
        min_h = min(r1[1], r2[1])
        area_r1 = r1[0] * r1[1]
        area_r2 = r2[0] * r2[1]

        intersect = min_w * min_h
        union = area_r1 + area_r2 - intersect

        return intersect / union

    def get_active_anchors(self,roi, anchors):
        indxs = []
        iou_max, index_max = 0, 0
       # print(roi)
        for i, a in enumerate(anchors):
           # print(roi)
            iou = self.iou_wh(roi[3:], a)
           # print(iou)
            if iou > 0.7:
                indxs.append(i)
               # print(iou)
            if iou > iou_max:
                iou_max, index_max = iou, i

        if len(indxs) == 0:
            indxs.append(index_max)
       # print(iou_max,index_max)
        return indxs

    def get_grid_cell(self,roi):
        x_center = roi[1]
        y_center = roi[2]

        grid_x = int(x_center *float(self.config.grid))
        grid_y = int(y_center *float(self.config.grid))

        return grid_x, grid_y

    def roi2label(self,roi, anchor):
        x_center = roi[1]
        y_center = roi[2]

        grid_x = x_center * float(self.config.grid)
        grid_y = y_center * float(self.config.grid)

        grid_x_offset = grid_x - int(grid_x)
        grid_y_offset = grid_y - int(grid_y)

        roi_w_scale = math.log(roi[3] / (anchor[0]/self.config.grid))
        roi_h_scale = math.log(roi[4] / (anchor[1]/self.config.grid))

        label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]
        #print(label)
        return label
    def convert_ground_truth(self,batch_y):

        label = np.zeros([self.config.batch_size, self.config.grid, self.config.grid, self.config.num, 6], dtype=np.float32)
        image_list = []
        for i, label in enumerate(batch_y):
            label_list = []
            for j, v in enumerate(label.decode().splitlines()):
                label_list.append([float(vv) for vv in v.split(' ')])
            image_list.append(label_list)
    #    print(image_list)
        anchors = self.config.anchors
        anchor_list = []
        for i in range(0, len(anchors), 2):
            anchor_list.append([anchors[i] / self.config.grid, anchors[i + 1] / self.config.grid])
      #  print(anchor_list)
        for image_index in range(len(image_list)):  # every image
            for label_index in range(len(image_list[image_index])):  # every label
                # print(image_list[image_index][label_index])
                activate_anchor_index = self.get_active_anchors(image_list[image_index][label_index], anchor_list)
                print(activate_anchor_index)
                grid_x, grid_y = self.get_grid_cell(image_list[image_index][label_index])
                for index in activate_anchor_index:
                    anchor_label = self.roi2label(image_list[image_index][label_index], anchor_list[index])
                    print(type(image_index),type(grid_x),type(grid_y),type(index),type(image_index),type(label_index))
                    cls = image_list[image_index][label_index][0]
                    label[image_index, grid_x, grid_y, index] = np.concatenate((anchor_label, [int(cls)], [1.0]))
        return label

    def train_epoch(self):

        self.sess.run(self.dataset.init_itorator())
        r = int(self.dataset.dataset_size/self.config.batch_size)

        loop = tqdm(range(r))
        losses = []
        loss_xy_es=[]
        loss_wh_es=[]
        loss_obj_es=[]
        loss_no_obj_es=[]
        loss_c_es=[]
        accs = []
        cur_it = self.model.global_step_tensor.eval(self.sess)
        for _ in loop:
           # try:
            loss, loss_xy, loss_wh, loss_obj, loss_no_obj, loss_c = self.train_step()
            #print(loss, loss_xy, loss_wh, loss_obj, loss_no_obj, loss_c)
            #print("loss:%f"%loss)

           # self.logger.summarize(cur_it, summaries_dict=summaries_dict)
            losses.append(loss)
            loss_xy_es.append(loss_xy*self.config.lambda_coord)
            loss_wh_es.append(loss_wh*self.config.lambda_coord)
            loss_obj_es.append(loss_obj)
            loss_no_obj_es.append(loss_no_obj*self.config.lambda_no_obj)
            loss_c_es.append(loss_c)
           # except tf.errors.OutOfRangeError:
            #    break
        loss = np.mean(losses)
        loss_xy = np.mean(loss_xy_es)
        loss_wh = np.mean(loss_wh_es)
        loss_obj = np.mean(loss_obj_es)
        loss_no_obj = np.mean(loss_no_obj_es)
        loss_c = np.mean(loss_c_es)

        summaries_dict = {
            'loss': loss,
            'loss_xy': loss_xy,
            'loss_wh': loss_wh,
            'loss_obj': loss_obj,
            'loss_no_obj': loss_no_obj,
            'loss_c': loss_c
        }



        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
      #  self.model.save(self.sess)

    def train_step(self):
        batch_x,batch_y = self.sess.run(self.dataset.next_batch())
        #TODO :  decode labels to the shape like predict tensor
        label = np.zeros([self.config.batch_size, self.config.grid, self.config.grid, self.config.num, 6],
                         dtype=np.float32)
        image_list = []
        for i, label__ in enumerate(batch_y):
            label_list = []
            for j, v in enumerate(label__.decode().splitlines()):
                label_list.append([float(vv) for vv in v.split(' ')])
            image_list.append(label_list)
        #    print(image_list)
        anchors = self.config.anchors

        anchor_list = []
        for i in range(0, len(anchors), 2):
            anchor_list.append([anchors[i] / self.config.grid, anchors[i + 1] / self.config.grid])
       # print(anchor_list)
        for image_index in range(len(image_list)):  # every image
            for label_index in range(len(image_list[image_index])):  # every label
                # print(image_list[image_index][label_index])
                #print(image_list[image_index][label_index])
                activate_anchor_index = self.get_active_anchors(image_list[image_index][label_index], anchor_list)
                #print(activate_anchor_index)
                grid_x, grid_y = self.get_grid_cell(image_list[image_index][label_index])
                for index in activate_anchor_index:
                    anchor_label = self.roi2label(image_list[image_index][label_index], anchor_list[index])
                    #print(type(image_index), type(grid_x), type(grid_y), type(index), type(image_index),type(label_index))
                  #  cls = image_list[image_index][label_index][0]
                  #  temp = np.concatenate([anchor_label,[int(cls)]])
                 #   label_=np.concatenate([temp,[1]])
                 #   label[image_index, grid_x, grid_y, index]=label_
                    label[image_index, grid_y, grid_x, index] = np.concatenate((anchor_label, [int(image_list[image_index][label_index][0])], [1.0]))
       # return label
        #batch_y_ = self.convert_ground_truth(batch_y)
        feed_dict = {self.model.istraining:True,self.model.inputs: batch_x, self.model.ground_truth: label}
        _, loss ,loss_xy,loss_wh,loss_obj,loss_no_obj,loss_c= self.sess.run([self.model.train_step, self.model.loss,self.model.loss_xy,self.model.loss_wh,self.model.loss_obj,self.model.loss_no_obj,self.model.loss_c],feed_dict=feed_dict)

        return loss,loss_xy,loss_wh,loss_obj,loss_no_obj,loss_c