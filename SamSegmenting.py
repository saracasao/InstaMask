import os
import cv2
import copy
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from visualization_utils import show_instructions, get_mask_img
from Loader import ImagesLoader

"""
INSTRUCTIONS TO USE InstaMask
Tool for labeling challenging objects with SAM. 
1. To introduce the label of the object that you will label press N 
2. In the terminal the code ask you for the label (should be a number)
3. Introduce label and press enter
4. Start labeling clicking on the object:
    4.1 Right button mouse = the pixel belongs to the object
    4.2 Press scroll wheel = the pixel does NOT belong to the object (if SAM includes some parts that are not from the object)
    4.3 Left button mouse = the mask is correct and you want to save it (sometimes opencv open a list of options, click again to save the mask). 
        When the mask has been saved, a message is shown in the terminal with the label and the path where the mask has been saved.
        If you don't click the left button mouse and continue to the next image or label, the mask is not going to be save

Different tools:
-> if you want to define a new label for the same image -> press N again, the code ask you again for the new label (steps 3. and 4.)
-> if you want to label the next image -> press D (you can change this letter in line 313)
-> if you want to finish the process press Esc
-> if you make a mistake during the labeling you can start the image again pressing C 
-> if you want to clean the images already preprocessed but not whose masks are not yet in the json:
   - images_processed.txt: save the key of the images that you have finished to work with, i.e., you must press D to save the key of the current image
"""

# GLOBAL VARIABLES
clicks = list()
label_clicks = list()
signal, mask_ok = False, False


def mouse_callback(event, x, y, flags, param):
    global signal
    global mask_ok
    global clicks
    global label_clicks

    #middle-bottom-click event value is 2
    if event == cv2.EVENT_RBUTTONDOWN:
        #store the coordinates of the right-click event
        mask_ok = True
        signal = True
    #this just verifies that the mouse data is being collected
    #you probably want to remove this later
    if event == cv2.EVENT_MBUTTONDOWN:
        clicks.append([x, y])
        label_clicks.append(0)
        signal = True
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append([x, y])
        label_clicks.append(1)
        signal = True


def save_mask_process(bool_mask, label, dir, n_masks, masks_path):
    folder_img = dir.split('/')[-4:-1]
    folder_img = os.path.join(*folder_img)

    m_path = masks_path + folder_img
    if not os.path.exists(m_path):
        os.makedirs(m_path)

    key_img = Path(dir).stem
    label_str = str(label)
    label_str = label_str.zfill(4)

    str_idx_mask = str(n_masks)
    str_idx_mask = str_idx_mask.zfill(4)

    name_mask = 'Img_' + key_img + '_L' + label_str + '_N' + str_idx_mask
    final_path = m_path + '/' + name_mask + '.png'
    cv2.imwrite(final_path, bool_mask * 255)

    n_masks += 1
    print('Masks labeled as {} save in {}'. format(label, final_path))
    return n_masks


def main(images_path, masks_path):
    global signal
    global mask_ok
    global clicks
    global label_clicks

    images_info = ImagesLoader(images_path,
                               masks_path,
                               clean_json_images=False,
                               load_only_json_images=False,
                               clean_processed_images=False,
                               path_file_annotations='./data/annotations_test.json')
    n_masks = images_info.n_masks_output_folder

    # SAM CONFIGURATION
    device = "cuda"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](
        checkpoint='/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth')
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print('Total images to label {}'.format(len(images_info.dir_images_rgb)))
    for i, dir_img in enumerate(images_info.dir_images_rgb):
        print('N {} Image {}'.format(i, dir_img))
        image = cv2.imread(dir_img)
        height, width, _ = image.shape

        # Load image in SAM
        predictor.set_image(image)

        # set mouse callback function for window
        window_name = str(Path(dir_img).stem)
        cv2.namedWindow(winname=window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        # USE TO LABEL ONLINE
        raw_image = copy.deepcopy(image)

        img_to_show = show_instructions(image)
        img_to_show = image.copy()

        img_processed = False
        # Interactive segmenting process
        while True:
            mask_ok = False
            mask_reference = cv2.hconcat([img_to_show, raw_image])
            cv2.imshow(window_name, mask_reference)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: # Esc = end process
                break
            elif k == ord('n'): # N = new label
                img_to_show = raw_image.copy()
                print('Label of the next mask?')
                label = int(input())
            elif k == ord('c'): # C = clean segmentation process
                clicks = list()
                label_clicks = list()
                img_to_show = copy.deepcopy(raw_image)
            elif k == ord('d'): # D = next image
                img_processed = True
                images_info.file_images_processed.write(str(Path(dir_img).stem) + '\n')
                break

            if signal and not mask_ok:
                clicks_arr = np.array(clicks)
                label_clicks_arr = np.array(label_clicks)

                masks, scores, logits = predictor.predict(
                    point_coords=clicks_arr,
                    point_labels=label_clicks_arr,
                    multimask_output=True,
                )

                # Select the mask with the highest score
                scores = list(scores)
                idx_mask_selected = scores.index(max(scores))

                # Show final mask
                bool_mask = masks[idx_mask_selected]
                mask_bgr = get_mask_img(bool_mask, label, images_info)
                img_to_show = cv2.addWeighted(raw_image, 1, mask_bgr, 0.9, 1) # raw_image

            elif signal and mask_ok:
                # Clean info for the new mask
                clicks = list()
                label_clicks = list()
                n_masks = save_mask_process(bool_mask, label, dir_img, n_masks, masks_path)
                img_to_show = copy.deepcopy(raw_image)
            signal, mask_ok = False, False

        if k == 27:
            if img_processed:
                images_info.file_images_processed.write(str(Path(dir_img).stem) + '\n')
            images_info.file_images_processed.close()
            break
        cv2.destroyAllWindows()
    images_info.file_images_processed.close()


if __name__ == "__main__":
    # Path images to be annotated
    path_images_to_segment = './data/'

    # Path to save the masks
    path_save_masks = './data_masks/'

    main(path_images_to_segment, path_save_masks)