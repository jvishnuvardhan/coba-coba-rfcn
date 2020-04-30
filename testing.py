import os
from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config

from KerasRFCN.Utils import Dataset, parseLabel, parseLabelDetection
import numpy as np
import pickle
from PIL import Image
import json
import tensorflow
import keras.backend as K


class DeepDriveDataset(Dataset):
    # count - int, images in the dataset
    def initDB(self, annotation, bddDir, count, start=0):
        self.start = start

        # image_dir = "/content/bdd100k/"
        all_images, classesCount, classesMapping = parseLabelDetection(
            annotation)
        self.classes = {}
        # Add classes (k class, c index) g bs pake class karna reference
        for k, c in classesMapping.items():
            self.add_class("Driving", c, k)
            self.classes[c] = k

        print('add image')
        for k, item in enumerate(all_images[start:count+start]):
            self.add_image(source="Driving", image_id=k,
                           filename=item['name'], width=1280, height=720, bboxes=item['bbox'])
        print("finish add image")
        self.rootpath = bddDir

    # read image from file and get the
    def load_image(self, image_id):
        info = self.image_info[image_id]
        # tempImg = image.img_to_array( image.load_img(info['path']) )
        print(info["filename"])
        tempImg = np.array(Image.open(
            os.path.join(self.rootpath, info['filename'])))
        return tempImg

    def get_keys(self, d, value):
        return [k for k, v in d.items() if v == value]

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        bboxes = []
        labels = []

        for item in info['bboxes']:
            bboxes.append((item['y1'], item['x1'], item['y2'], item['x2']))
            label_key = self.get_keys(self.classes, item['category'])
            if len(label_key) == 0:
                continue
            labels.extend(label_key)
        return np.array(bboxes), np.array(labels)


def main():
    # tensorflow.compat.v1.disable_resource_variables()
    # with tensorflow.compat.v1.Session() as ses:
    # ses = tensorflow.compat.v1.Session()
    # tensorflow.debugging.set_log_device_placement(True)
    ROOT_DIR = os.getcwd()
    # # inisialisasi config
    config = Config()

    # inisialisasi dataset training
    dataset_train = DeepDriveDataset()
    dataset_train.initDB(
        "d:/bdd100k/detection_image_train.json", "D:/Workspace/College/Semester 8/Tugas Akhir/codeR-FCN/bdd100k/images/100k/train", 70000)
    dataset_train.prepare()
    print(dataset_train.image_ids, "masuk d ong")

    # Validation dataset
    dataset_val = DeepDriveDataset()
    dataset_val.initDB(
        "d:/bdd100k/detection_image_val.json", "D:/Workspace/College/Semester 8/Tugas Akhir/codeR-FCN/bdd100k/images/100k/val", 10000)
    dataset_val.prepare()
    # siapin model rfcn untuk training
    model = RFCN_Model(mode="training",
                       model_dir=os.path.join(ROOT_DIR, "ModelData"))
    # init = tf.global_variables_initializer()
    try:
        model_path = model.find_last()[1]
        print(model_path)
        if model_path is not None:
            model.load_weights(model_path, by_name=True)
    except Exception as e:
        print(e)
        print("No checkpoint founded")

    # K.manual_variable_initialization(True)
    model.save("D:/weight.h5")
    # tensorflow.compat.v1.enable_resource_variables()
    # training begin
    # stage 1
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')
    model.save("D:/weight.h5")
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')
    model.save("D:/weight.h5")
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers='all')
    model.save("D:/weight.h5")
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=240,
                layers='all')
    model.save("D:/weight.h5")


if __name__ == "__main__":
    # with tensorflow.compat.v1.Session() as ses:
    #     ses.run(tensorflow.global_variables_initializer())
    main()
    # main()
