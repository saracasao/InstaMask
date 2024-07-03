# InstaMask 

Official tool for labeling SpectralWaste dataset with the assistance of SAM ([SAM web](https://segment-anything.com/) , [SAM repo](https://github.com/facebookresearch/segment-anything)).


## Installation

```
python3.8 -m venv "insta_mask"
git clone https://github.com/saracasao/InstaMask.git
cd InstaMask
pip install requirements.txt
```

## Quick guide 

The python file _SamSegmenting.py_ open an interactive labeling process based on the point prompt of SAM. It is necessary to define the directories of the images to be labeled and where to save the generated masks:

```python
if __name__ == "__main__":
    # Path images to be annotated
    path_images_to_segment = './data/'

    # Path to save the masks
    path_save_masks = './data_masks/'

    main(path_images_to_segment, path_save_masks)
```

When _SamSegmenting.py_ file is lunched, the same image appears twice. The labeling process must be made in the left image while the right one is only to help as a refence during the labeling process.

Instructions:
1. To introduce the label of the object press N 
2. In the terminal the code ask you for the label (should be a number)
3. Introduce label and press enter
4. Start the labeling process clicking on the object:
    4.1 Left button mouse = the pixel belongs to the object
    4.2 Press scroll wheel = the pixel does NOT belong to the object
    4.3 Right button mouse = the mask is correct and you want to save it (sometimes opencv open a list of options, click again to save the mask). 
        * When the mask has been saved, a message is shown in the terminal with the label and the path where the mask has been saved.
        * If you don't click the left button mouse and continue to the next image or label, the mask is not saved

Different tools:
* if you want to define a new label for the same image -> press N again, the code ask you again for the new label (steps 3. and 4.)
* if you want to label the next image -> press D
* if you want to finish the process press Esc
* if you make a mistake during the labeling you can start the image again pressing C 
* if you want to clean the images already preprocessed `clean_processed_images=True`:
  _images_processed.txt_ save the key of the images that you have finished to work with, i.e., you must press D to save the key of the current image

The output is a binary .png image with the following structure: `Img_ + key_img + _L + label_str + _N + str_idx_mask + '.png`

## SpectralWaste Dataset 

To include the generated masks into the spectralwaste annotations .json file or create a new .json file use _masks_to_json.py_
