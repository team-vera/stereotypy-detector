import sys
import os
sys.path.append("..")
sys.path.append("../yolov5")
from detection.detection_utils.yolo_wrapper import Yolo
from detection.detection_utils.detector import Detector
from utilities.drawing import draw_bboxes_raw
from utilities.bbox_utils import crop_bboxes_raw
from identification.identification_utils.data_loading import get_resize_transform
from utilities.enums import POLAR_NAMES
import numpy as np
import cv2
import argparse
import torch
import time


def process_img_(img: np.ndarray, det_model: Yolo, ident_model: torch.nn.Module = None, device: str = "cpu"):

    bboxes = det_model.detect(img[np.newaxis, :, :, ::-1])[0]

    cropped_images = crop_bboxes_raw(img, bboxes)

    if ident_model is not None and len(cropped_images) > 0:
        transform = get_resize_transform()
        cropped_images = torch.stack(
            [transform(i[:, :, ::-1].copy()) for i in cropped_images]).to(device)

        ident_pred = ident_model(cropped_images).cpu().detach().numpy()

        for i, (bbox, pred) in enumerate(zip(bboxes, ident_pred)):

            pred_thresh = np.where(pred > 0.6, 1, 0)

            if 1 not in pred_thresh:
                class_name = "Unknown"
            else:
                class_name = POLAR_NAMES[np.argmax(pred)]

            bboxes[i] = [*bbox[:5], class_name, np.max(pred)]

    img = draw_bboxes_raw(img, bboxes, "BGR")

    return img

def process_img(img: np.ndarray, detector: Detector):
    
    bboxes = detector.detect(img[np.newaxis, :, :, ::-1])[0]
    
    img = draw_bboxes_raw(img, bboxes, "BGR")
    
    return img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--det_weights",
                        type=str,
                        required=True,
                        help="Path to trained detection weights")
    parser.add_argument("--ident_weights",
                        type=str,
                        default=None,
                        help="Path to trained identification weights")
    parser.add_argument("--input_video",
                        type=str,
                        default=None,
                        help="Path to video to detect on")
    parser.add_argument("--input_image",
                        type=str,
                        default=None,
                        help="Image to detect on or path to directory with only images")
    parser.add_argument("--img_size",
                        type=int,
                        nargs=2,
                        default=(960, 540),
                        help="Size to use for displaying (width, height)")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    detector = Detector(det_weights=args.det_weights,
                        device=device,
                        ident_weights=args.ident_weights,
                        det_conf=0.5,
                        ident_conf=0.6)

    if args.input_video is not None:
        cap = cv2.VideoCapture(args.input_video)

        while 1:
            start_time = time.time()
            ret, img = cap.read()

            img = cv2.resize(img, args.img_size)

            img = process_img(img, detector)
            
            cv2.imshow("Polar Bears!!!", img)
            k = cv2.waitKey(10)
            if k == ord("q"):
                cv2.destroyWindow("Polar Bears!!!")
                break
            elif k == ord(" "):
                while 1:
                    k = cv2.waitKey(0)
                    if k == ord(" "):
                        break
                    
            end_time = time.time()
            print("FPS: {:.1f}   ".format(1 / (end_time - start_time)),
                  end="\r")

        print()
    elif args.input_image is not None:
        if os.path.isdir(args.input_image):
            image_paths = os.listdir(args.input_image)
        else:
            image_paths = [args.input_image, ]
            
        image_paths.sort()

        i = 0

        while 1:
            img = cv2.imread(os.path.join(args.input_image, image_paths[i]))
            img = cv2.resize(img, args.img_size)

            img = process_img(img, detector)

            cv2.imshow("Polar Bears!!!", img)

            do_break = False
            while 1:
                k = cv2.waitKey(0)
                if k == ord("q"):
                    do_break = True
                    break
                elif k == ord(" "):
                    i += 1
                    break
                elif k == ord("\b"):
                    i -= 1
                    break
  
            i %= len(image_paths)

            if do_break:
                break

        cv2.destroyWindow("Polar Bears!!!")
    else:
        raise NotImplementedError("No valid input was given")


if __name__ == "__main__":
    main()
