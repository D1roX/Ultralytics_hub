import os
import numpy as np
import cv2
from ultralytics import YOLO, checks, hub


def train():
    checks()

    hub.login('69b4fdb1574bc71a0fb9f5870ff26e9dddd008fbb3')

    model = YOLO('https://hub.ultralytics.com/models/ZmuNcMLjCdKtP5EoSIBK')

    model.train()


def detect():
    model = YOLO(
        'P:\\PyCharmProjects\\ultralytics_hub_train\\runs\\detect\\train13\\weights\\best.pt')
    model.info()
    model.cuda()
    result = model.predict('F:\\Work\\NavigationSystem\\TestVideo\\231015_тестовый_пролет\\DJI_0063_2.mp4',
                           save=True,
                           stream=False,
                           device='0',
                           conf=0.6,
                           line_width=2,
                           iou=0.25)


def concat_results():
    paths = [
        'runs/detect/predict16',
        'runs/detect/predict17',
        'runs/detect/predict18'
    ]
    for filename1, filename2, filename3 in zip(os.listdir(paths[0]), os.listdir(paths[1]), os.listdir(paths[2])):
        im1_path = os.path.join(paths[0], filename1)
        im2_path = os.path.join(paths[1], filename2)
        im3_path = os.path.join(paths[2], filename3)
        im1 = cv2.imread(im1_path)
        im2 = cv2.imread(im2_path)
        im3 = cv2.imread(im3_path)
        size = 850
        im1 = cv2.resize(im1, (size, size))
        im2 = cv2.resize(im2, (size, size))
        im3 = cv2.resize(im3, (size, size))
        vis = np.concatenate((im1, im2, im3), axis=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 255, 255)
        thickness = 1
        lineType = 2
        cv2.putText(vis, 'v5s',
                    (50, 50),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(vis, 'v8s',
                    (50 + size, 50),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(vis, 'v8s_pred',
                    (50 + 2*size, 50),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        # Display the image
        cv2.imshow("img", vis)
        cv2.waitKey(0)


def vs():
    dir = 'test_imgs2'
    i = 0
    i += 1
    models = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        models.append((filename, YOLO(path).cuda()))
        print(filename)

    img_dir = 'test_imgs'
    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)
        for name, model in models:
            model.predict(path,
                          save=True,
                          device='0',
                          hide_labels=True,
                          hide_conf=True,
                          line_width=2,
                          iou=0.25)


def detect_photo():
    path = 'P:\\PyCharmProjects\\ultralytics_hub_train\\runs\\detect\\train7\\weights\\best.pt'
    path = 'best.pt'
    model = YOLO(path).cuda()
    im_path = 'test_imgs2'
    for filename in os.listdir('test_imgs2'):
        full_im_path = os.path.join(im_path, filename)
        model.predict(full_im_path,
                      save=True,
                      device='0',
                      conf=0.6,
                      line_width=2,
                      iou=0.25)


if __name__ == '__main__':
    # train()
    # detect()
    # concat_results()
    detect_photo()
