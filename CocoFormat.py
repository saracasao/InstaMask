import os
import re
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import date
from torchvision import transforms
from utils import binary_mask_to_bbox, mask_to_rle


class CocoFormatSPECTRALWASTE:
    def __init__(self, dir_dataset, dir_masks, dir_previous_annotations=None):
        """COCOFormatSEPCTRALWASTE transform binary mask images into COCOFormat for saved them in a json
           dir_dataset: str path dataset
           dir_masks: str path generated masks with InstaMask
           dir_previous_annotations: str path json with the existing annotations"""

        self.dir_data = dir_dataset
        self.dir_masks = dir_masks

        # ---------------------General label references in the annotated data----------------------------------------
        self.SPECTRALWASTE_NUM_LABELS = 7
        self.SPECTRALWASTE_LABELS = {'background': 0,
                                     'film': 1,
                                     'basket': 2,
                                     'carboard': 3,
                                     'video_tape': 4,
                                     'filament': 5,
                                     'bag': 6}

        self.SPECTRALWASTE_LABELS_COLORS = {0: '#000000',
                                            1: '#daf706',
                                            2: '#33ddff',
                                            3: '#3432dd',
                                            4: '#ca98c3',
                                            5: '#3068df',
                                            6: '#ffa500'}

        # ----------------------Required dictionaries for COCO labeling format------------------------------------
        self.info = {'description': 'SPECTRALWASTE Dataset',
                     'url': '',
                     'version': '',
                     'year': '2023',
                     'date_created': ''}

        self.categories = [{'id': 0, 'name': 'background'},  # background = the rest of the object
                           {'id': 1, 'name': 'film'},
                           {'id': 2, 'name': 'basket'},
                           {'id': 3, 'name': 'carboard'},
                           {'id': 4, 'name': 'video_tape'},
                           {'id': 5, 'name': 'filament'},
                           {'id': 6, 'name': 'bag'}]

        self.licenses = [{'id': 0,
                          'name': 'No known copyright restrictions',
                          'url': ''}]

        self.convert_tensor = transforms.PILToTensor()
        self.images = []
        self.images_keys = ['id', 'file_name', 'width', 'height', 'data_captured']

        self.annotation_id = 0
        self.annotations = []
        self.annotations_keys = ['id', # annotation id -> each annotation is unique
                                 'image_id', # id of the labeled image
                                 'category_id', # int label (defined in category dict)
                                 'segmentation', # coordinates of the polynom
                                 'bbox'] # bbox coordinates [xmin, ymin, width, height]
        if dir_previous_annotations is not None:
            self.dir_annotation_file = dir_previous_annotations
            self.previous_annotations = self.set_dataset_labels()
        else:
            self.init_new_annotations()

    def init_new_annotations(self):
        """Update the last modification of the json annotations file"""
        current_date = date.today()
        self.info['date_created'] = current_date.strftime("%Y/%m/%d/")

    def set_dataset_labels(self):
        """Load data annotations contained in the existing json"""
        with open(self.dir_annotation_file) as f:
            data = json.load(f)

        # Update version
        current_date = date.today()
        data['info']['version'] = current_date.strftime("%Y/%m/%d/")
        self.images = data['images']
        self.annotations = data['annotations']
        self.annotation_id = self.annotations[-1]['id'] + 1
        return data

    def get_image_id(self):
        images_id = [f['id'] for f in self.images]
        images_id = sorted(images_id)
        return images_id

    def get_files_dir_from_path(self):
        """Load the masks and look for the corresponding images in the global folder of the dataset"""
        # Get all the new masks
        dir_masks_png = list(Path(self.dir_masks).rglob("*.png")) # check extension of the masks
        dir_masks_jpg = list(Path(self.dir_masks).rglob("*.jpg")) # check extension of the masks

        list_dir_masks = dir_masks_png + dir_masks_jpg

        # Get all the images corresponding to the masks
        list_dir_images = []
        for d_mask in list_dir_masks:
            d_mask_str = str(d_mask)
            img_id = re.search('Img_(.*)_L', d_mask_str).group(1)
            subfolder = os.path.join(*d_mask_str.split('/')[-4:-1])
            dir_image_png = self.dir_data + subfolder + '/' + img_id + '.jpg'
            assert os.path.isfile(dir_image_png)
            list_dir_images.append(Path(dir_image_png))

        # Get info images and masks as str
        info_image_to_label = [(str(f), str(f.stem)) for i, f in enumerate(list_dir_images)]
        list_dir_masks = list(map(str, list_dir_masks))

        return sorted(info_image_to_label), sorted(list_dir_masks)

    def get_name_files_masks_as_ref(self):
        """Return info_images: list([tuple(path_img0,key_image0), ...]
                  list_dir_masks: list([path_mask0, ...]"""
        info_images, list_dir_masks = self.get_files_dir_from_path()
        return info_images, list_dir_masks

    def create_image_info(self, image_id, file_name, width, height, hyper):
        img_dir_split    = file_name.split('/')
        img_dir_standard = img_dir_split[img_dir_split.index('data'):]
        img_dir_standard = os.path.join(*img_dir_standard)

        if hyper:
            extension = Path(img_dir_standard).suffix
            img_dir_standard = img_dir_standard.replace('rgb', 'hyper')
            img_dir_standard = img_dir_standard.replace(extension,'.tiff')

        image_info = {"id": image_id,
                      "file_name": img_dir_standard,
                      "width": width,
                      "height": height,
                      "date_captured": self.info['date_created'],
                      "license": 0}
        return image_info

    def create_annotation_info(self, mask, name_file_mask, image_id):
        """Translate binary mask to rle format
           mask: PIL file
           name_file_mask: str path mask
           image_id: str key image labeled"""

        # From mask to rle
        mask_tensor = self.convert_tensor(mask)
        segmentation = mask_to_rle(mask_tensor)

        bbox = binary_mask_to_bbox(mask)

        category_str = re.search('_L(.*)_N', name_file_mask).group(1)
        category_id  = int(category_str)
        assert category_id in self.SPECTRALWASTE_LABELS.values()

        # Save new info
        annotation_info = {'id': self.annotation_id,
                           'image_id': image_id,
                           'category_id': category_id,
                           'segmentation': segmentation,
                           'bbox': bbox}
        self.annotation_id += 1
        return annotation_info

    def generate_new_coco_labels(self, hyper=False):
        """ Load masks and transform the binary images into labels in the COCO format
        hyper Bool: indicate if the labeled images belongs to the hyperspectral set"""
        info_images, dir_masks = self.get_name_files_masks_as_ref()
        dir_images, name_images_wo_ext = zip(*info_images)

        for i, d_mask in enumerate(dir_masks):
            img_id = re.search('Img_(.*)_L', d_mask).group(1)
            idx_name_image = name_images_wo_ext.index(img_id)
            dir_img = dir_images[idx_name_image]

            mask = Image.open(d_mask)
            assert len(list(np.unique(mask))) == 2 or len(list(np.unique(mask))) == 1, "Error gray values in mask instead of binary"
            images_id = self.get_image_id()
            if img_id not in images_id:
                image_info = self.create_image_info(img_id, dir_img, mask.size[0], mask.size[1], hyper)
                self.images.append(image_info)

                annotation_info = self.create_annotation_info(mask, d_mask, img_id)
                self.annotations.append(annotation_info)
            else:
                annotation_info = self.create_annotation_info(mask, d_mask, img_id)
                self.annotations.append(annotation_info)

    def save_annotations(self, dir_file):
        coco_annotations = {'info' : self.info,
                            'licenses': self.licenses,
                            'categories': self.categories,
                            'images': self.images,
                            'annotations': self.annotations}

        with open(dir_file, 'w') as outfile:
            json.dump(coco_annotations, outfile)
        print('ANNOTATIONS SAVE IN:', dir_file)

