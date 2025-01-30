# import the necessary packages
import argparse
import cv2
from inky.auto import auto
import numpy as np
#from matplotlib import pyplot as plt
from PIL import Image

def load_image(image_path):
    return cv2.imread(image_path)

def save_image(image_path, image):
    cv2.imwrite(image_path, image)

def crop(image, disp_w, disp_h, intelligent=True):
    # Intelligently resize and crop image to display proportions.
    # Largest crop shifted towards maximum saliency. 

    img_h, img_w, img_c = image.shape
    print(f"Input WxH: {img_w} x {img_h}")

    img_aspect = img_w / img_h
    disp_aspect = disp_w / disp_h

    print(f"Image aspect ratio {img_aspect} ({img_w} x {img_h})")
    print(f"Display aspect ratio {disp_aspect} ({disp_w} x {disp_h})")

    if img_aspect < disp_aspect:
        # scale width, crop height.
        resize =(disp_w, int(disp_w / img_aspect))
    else:
        # scale height, crop width
        resize = (int(disp_h * img_aspect), disp_h)

    print(f"Resizing to {resize}")
    image = cv2.resize(image, resize)
    img_h, img_w, img_c = image.shape

    # Cropping
    x_off = int((img_w - disp_w) / 2)
    y_off = int((img_h - disp_h) / 2)
    assert x_off == 0 or y_off == 0, "My logic is broken"

    if intelligent:

        # Initialize OpenCV's static saliency spectral residual detector and
        # compute the saliency map
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        if not x_off:
            # Cropping height
            vert = np.max(saliencyMap, axis=1)
            vert = np.convolve(vert, np.ones(64)/64, "same")
             
            sal_centre = int(np.argmax(vert))
            img_centre = int(img_h / 2)
            shift_y = max(min(sal_centre - img_centre, y_off), -y_off)
            y_off += shift_y
        else:
            # Cropping width
            horiz = np.max(saliencyMap, axis=0)
            horiz  = np.convolve(horiz, np.ones(64)/64, "same")
            sal_centre = int(np.argmax(horiz))
            img_centre = int(img_w / 2)
            shift_x = max(min(sal_centre - img_centre, x_off), -x_off)
            x_off += shift_x

    image = image[y_off:y_off + disp_h,
                  x_off:x_off + disp_w]

    img_h, img_w, img_c = image.shape
    print(f"Cropped WxH: {img_w} x {img_h}")
    return image

def display(inky, image, saturation=1.0):
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    

    inky.set_image(Image.fromarray(image), saturation=saturation)
    inky.show()


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image", 
        help="input image")
    ap.add_argument("-o", "--output", default="",
        help="name to save cropped display image if provided")
    ap.add_argument("-p", "--portrait", action="store_true",
                    default=False, help="Portrait orientation")
    ap.add_argument("-c", "--centre_crop", action="store_true",
                    default=False, help="Simple centre cropping")
    ap.add_argument("-r", "--resize_only", action="store_true",
                    default=False, help="Simply resize image to display ignoring aspect ratio")
    ap.add_argument("-s", "--simulate_display", action="store_true",
                    default=False, help="Do not interact with e-paper display")
    args = vars(ap.parse_args())

    simulate_display = args["simulate_display"]

    if simulate_display:
        disp_w, disp_h = (800, 480)
    else:
        inky = auto(ask_user=True, verbose=True)
        disp_w, disp_h = inky.resolution

    if args["portrait"]:
        disp_w, disp_h = disp_h, disp_w
    
    image = load_image(args["image"])
    if args["resize_only"]:
        resize = inky.resolution
        print(f"Resizing to {resize}")
        image = cv2.resize(image, resize)
    else:
        image = crop(image, disp_w, disp_h, args["centre_crop"]==False)

    if not simulate_display:
        display(inky, image)

    if args["output"]:
        save_image(args["output"], image)


