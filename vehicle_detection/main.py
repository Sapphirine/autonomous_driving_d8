import yolo_od as yolo1
import data
import numpy as np
import cv2

def adjust_channel_gamma(channel, gamma=1.):
    invGamma = 1.0 / np.absolute(gamma)
    table = (np.array([((i / 255.0) ** invGamma) * 255
                       for i in np.arange(0, 256)]).astype("uint8"))
    return cv2.LUT(channel, table)

def adjust_image_gamma(img, gamma=1.):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = adjust_channel_gamma(img[:, :, 2], gamma=gamma)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

if data.isVideo:
    fourcc = cv2.VideoWriter_fourcc(*  'WMV2') #'MJPG')
    filename = 'output_images/YOLO_projectvideo.mp4' # + data.video
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
    cam = cv2.VideoCapture(data.img_add)

yolo = yolo1.yolo(model="cfg/tiny-yolo-voc.cfg", chkpt="bin/tiny-yolo-voc.weights", threshold=0.12)

while(True):
    if data.isVideo:
        ret, image = cam.read()
        if ret == False:
            break
    else:
        image = cv2.imread(data.img_add, -1)
    gamma_img = adjust_image_gamma(image.copy(), 2)
    objs = yolo.find_object(gamma_img) # find the objects
    image = yolo.draw_box(image, objs, show_label=True) # add the detected objects to the window
    cv2.imshow('final', image)

    # wait for a user key interrupt then close all windows
    if data.isVideo:
        out.write(image)  # save image to video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imwrite('output_images/objects_' + data.image, image)
        cv2.waitKey(0)
        break

if data.isVideo:
    out.release()
    cam.release()

cv2.destroyAllWindows()
