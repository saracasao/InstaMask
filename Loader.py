import re
import os
import json
from pathlib import Path
import numpy as np


class Dataset:
    def __init__(self, path_json_annotations):
        self.path_json = path_json_annotations
        self.annotations = json.load(open(self.path_json , 'r'))
        self.image_info = self.annotations['images']
        self.key_images = [Path(p['file_name']).stem for p in self.image_info]
        self.annotations_info = self.annotations['annotations']
        self.colors_dict = self.get_colors()
        self.segmentations = None

    def load_segmentations(self, key_images_selection=None):
        # Init dictionary
        if key_images_selection is None:
            ann = {name_img: [] for name_img in self.key_images}
        else:
            ann = {name_img: [] for name_img in key_images_selection}

        # Get images ids with annotations
        ann_image_ids = [ann['image_id'] for ann in self.annotations_info]
        ann_image_ids = np.array(ann_image_ids)
        for name in self.key_images:
            if name in ann_image_ids:
                idx_ann = np.where(name == ann_image_ids)[0]
                for idx in idx_ann:
                    img_ann = self.annotations_info[idx]['segmentation']
                    label_ann = self.annotations_info[idx]['category_id']
                    info_ann = tuple((img_ann, label_ann))
                    ann[name].append(info_ann)
        self.segmentations = ann

    def get_colors(self):
        categories = self.annotations['categories']
        color_dict = {}
        for c_dict in categories:
            color_dict[c_dict['id']] = self.hex_to_rgba(c_dict['color'])
        return color_dict

    @staticmethod
    def hex_to_rgba(hex):
        if '#' in hex:
            hex = hex.replace('#', '')
        color = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
        # color = np.concatenate([np.array(color), np.array([0.6])], axis=0)
        return np.array(color)


class ImagesLoader:
    def __init__(self, dir_folder, output_folder, clean_json_images=False, load_only_json_images=False,
                 clean_processed_images=False, path_file_annotations=None):
        self.dataset = None
        self.dir_project = dir_folder
        self.output_folder = output_folder
        self.dir_images_rgb = self.load_rgb_images()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.n_masks_output_folder = 0
        else:
            self.dir_images_segmented, self.n_masks_output_folder = self.check_masks_folder()
        self.file_images_processed, self.old_key_images_preprocessed = self.load_images_processed()

        if path_file_annotations is not None:
            assert path_file_annotations is not None, "Annotation file None"
            self.dataset = Dataset(path_file_annotations)
            if clean_json_images:
                self.remove_json_images()
            elif load_only_json_images:
                self.dataset.load_segmentations()
                self.load_only_json_images()

        if clean_processed_images:
            self.clean_images_already_processed()

    def load_rgb_images(self):
        dir_images_jpg = list(Path(self.dir_project).rglob("*.jpg"))
        dir_images_png = list(Path(self.dir_project).rglob("*.png"))
        dir_images_rgb = dir_images_jpg + dir_images_png
        dir_images_rgb = list(map(str, dir_images_rgb))
        return dir_images_rgb

    def remove_json_images(self):
        # Load key images in the json
        key_images_annotated = self.dataset.key_images
        key_images_folder = [Path(p).stem for p in self.dir_images_rgb]

        # Delete images already annotated in the json
        key_images_to_annotate = sorted(list(set(key_images_folder) - set(key_images_annotated)))
        self.dir_images_rgb = [p for p in self.dir_images_rgb if Path(p).stem in key_images_to_annotate]

    def load_only_json_images(self):
        self.dir_images_rgb = [p for p in self.dir_images_rgb if Path(p).stem in self.dataset.key_images]

    def load_images_processed(self):
        old_key_images_preprocessed = []
        path_file = self.output_folder + 'images_processed.txt'
        if os.path.exists(path_file) and os.path.getsize(path_file) > 0:
            old_file_images_processed = open(path_file, "r")
            lines = [line.rstrip() for line in old_file_images_processed]

            file_images_processed = open(path_file, "w+")
            for l in lines:
                old_key_images_preprocessed.append(l)
                file_images_processed.write(l + '\n')
            old_key_images_preprocessed = np.unique(old_key_images_preprocessed)
            old_key_images_preprocessed = list(map(str, old_key_images_preprocessed))
        else:
            file_images_processed = open(path_file, "w+")
        return file_images_processed, old_key_images_preprocessed

    def check_masks_folder(self):
        dir_masks_jpg = list(Path(self.output_folder).rglob("*.jpg"))
        dir_masks_png = list(Path(self.output_folder).rglob("*.png"))
        dir_masks = dir_masks_jpg + dir_masks_png

        key_images_segmented = [re.search('Img_(.*)_L', str(d)).group(1) for d in dir_masks]
        key_images_segmented = list(np.unique(key_images_segmented))
        masks_number = [int(re.search('_N(.*).png', str(d)).group(1)) for d in dir_masks]
        if len(masks_number) > 0:
            n_masks = max(masks_number)
        else:
            n_masks = 0
        return key_images_segmented, n_masks

    def clean_images_already_processed(self):
        self.dir_images_rgb = [p for p in self.dir_images_rgb if Path(p).stem not in self.old_key_images_preprocessed]
