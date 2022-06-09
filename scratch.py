import glob
import os
import shutil
from venv import create

"""{'Clothes_hanger': 5, 'walkman': 11, 'fish': 15, 'usb_drive': 24,
    'violin': 28, 'toy_car': 31, 'insect': 35, 'fork': 37, 'bed': 42,
    'Photo_Frame': 44, 'table': 59, 'laptop': 64, 'can': 65, 'cup': 68,
    'cell_phone': 72, 'sword': 79, 'knife': 84, 'sofa': 86, 'vase': 107,
    'stool': 114, 'computer_monitor': 142, 'gun': 143, 'toy_plane': 155,
    'guitar': 175, 'bottle': 178, 'pen+pencil': 190, 'plants': 204,
    'figurines': 219, 'Lamp': 267, 'Chair': 389}"""

DATA_PATH = 'data'

top_5 = ['Chair', 'Lamp', 'figurines', 'plants', 'pen+pencil']  # 190
top_10 = ['gun', 'computer_monitor', 'toy_plane', 'guitar',
          'bottle', 'pen+pencil', 'plants', 'figurines', 'Lamp', 'Chair']  # 143

def create_test(top_n_list, top_n_str):
    for cls in top_n_list:
        move_count = 0
        for img_path in glob.iglob('%s/%s/train/%s/*/*' % (DATA_PATH, top_n_str, cls)):
            if not img_path.endswith('RGB.png'):
                continue
            if move_count >= 20:
                continue

            move_count += 1
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]

            if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test')):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls))
            if img_id not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test', cls)):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls, img_id))

            for file in glob.glob('%s/%s/train/%s/%s/%s_%s*' % (DATA_PATH, top_n_str, cls, img_id, img_var, img_id)):
                name = file.split('\\')[-1]
                shutil.move(file, '%s/%s/test/%s/%s/%s' % (DATA_PATH, top_n_str, cls, img_id, name))

def create_top_n(top_n_list, top_n_str):
    if top_n_str not in os.listdir(DATA_PATH):
        os.mkdir(os.path.join(DATA_PATH, top_n_str))
        os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train'))
        os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test'))
        
    for cls in top_n_list:
        n_img = 0
        if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'train')):
            os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train', cls))
        if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test')):
            os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls))
        for img_path in glob.iglob('%s/*/%s/*/*' % (DATA_PATH, cls)):
            if n_img >= 185:
                continue
            if img_path.endswith('RGB.png'):
                n_img += 1
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]

            if img_id not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'train', cls)):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train', cls, img_id))
            
            shutil.copyfile(img_path, os.path.join(DATA_PATH, top_n_str, 'train', cls, img_id, img_name))


def count():
    cls_list = []
    with open(os.path.join('data', 'cls.txt'), 'r') as f:
        file = f.readlines()
        for cls in file:
            # remove '\n' from string
            cls = cls[:-1]
            cls_list.append(cls)

    img_id_dict = {}
    for img_path in glob.iglob('%s/*/*/*/*' % 'data'):
        if not img_path.endswith('RGB.png'):
            continue
        
        img_cls = img_path.split('\\')[-3]
        # E.g. '<img_idx>_<img_id>_<img_type>.png'
        img_name = img_path.split('\\')[-1]
        img_var = img_name.split('_')[0]
        img_id = img_name.split('_')[1]
        img_id_with_var = img_var + '_' + img_id
        img_id_dict[img_id_with_var] = img_cls

    cls = list(img_id_dict.values())
    cls_dict = {}
    for i in range(30):
        cls_dict[cls_list[i]] = cls.count(cls_list[i])

    ordered_cls_dict = {k: v for k, v in sorted(cls_dict.items(), key=lambda item: item[1])}
    print(ordered_cls_dict)


def create_cls_txt(cls_list, file_path):
    with open(file_path, 'w') as f:
        for cls in cls_list:
            f.write(cls)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    #count()
    create_cls_txt(top_10, '%s/cls_top_10.txt' % DATA_PATH)
    #create_top_5()
    #create_test()