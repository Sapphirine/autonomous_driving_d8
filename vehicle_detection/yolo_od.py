import cv2
from darkflow.net.build import TFNet

class yolo():

    def __init__(self, model, chkpt, threshold):
        options = {"model": model, "load": chkpt, "threshold": threshold}
        self.tfnet = TFNet(options)

    def draw_box(self, image, objects, box_color=(0,0,255), show_label=False):
        img = image.copy()
        for obj in objects:
            pt1 = (obj['bottomright']['x'], obj['bottomright']['y'])
            pt2 = (obj['topleft']['x'], obj['topleft']['y'])
            cv2.rectangle(img, pt1, pt2, box_color, 4)
            if show_label is True:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, obj['label'], pt2, font, 0.75, (0,255,0), 2)
                #cv2.putText(img, obj['label'], pt2, 0.75, (0,255,0), 2)

        return img

    def find_object(self, image):
        return self.tfnet.return_predict(image)
