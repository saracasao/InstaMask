from CocoFormat import CocoFormatSPECTRALWASTE
from datetime import date


def main():
    current_date = date.today()
    current_date = current_date.strftime("%Y-%m-%d_")
    name_file = current_date + 'annotations_update.json'

    dir_images = './data/'
    dir_masks = './data_masks/'

    coco_labels = CocoFormatSPECTRALWASTE(dir_images, dir_masks, dir_previous_annotations='./data/annotations_test.json')
    coco_labels.init_new_annotations()
    coco_labels.generate_new_coco_labels()

    coco_labels.save_annotations('./data/' + name_file)


if __name__ == '__main__':
    main()

