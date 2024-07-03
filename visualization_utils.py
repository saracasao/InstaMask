import cv2
import numpy as np


def show_instructions(img):
    font_size = 20
    height, width, _ = img.shape

    center_w = int(width / 2)
    center_h = int(height / 3)

    # Text configuration
    text = 'Press N to introduce \n a new label in the terminal'
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_width, text_height = text_size[0], text_size[1]

    # Coordinates of the text to show
    init_coord = (center_w - int(text_width / 2), center_h - int(text_height/2))
    end_coord = (init_coord[0] + text_width, init_coord[1] + text_height)

    box_coord_init = (init_coord[0] - int(text_width * 0.05), init_coord[1] - int(text_height * 0.8))
    box_coord_end = (end_coord[0] + int(text_width * 0.05), end_coord[1] + int(text_height * 0.8))

    #Draw instructions
    img = cv2.rectangle(img, box_coord_init, box_coord_end, (127, 127, 127), -1)
    img = cv2.rectangle(img, box_coord_init, box_coord_end, (0, 0, 0), 2)
    cv2.putText(img, 'Press N to introduce a new label in the terminal', (init_coord[0], end_coord[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),  2)

    return img


def get_mask_img(bool_mask, label, data_info):
    if data_info.dataset is not None:
        colors_dict = data_info.dataset.colors_dict
        color = colors_dict[label]
    else:
        color = np.random.random(3) * 255
    h, w = bool_mask.shape[-2:]
    mask_image = bool_mask.reshape(h, w, 1) * color
    mask_rgb = mask_image.astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    return mask_bgr
