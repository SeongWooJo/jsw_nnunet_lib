import os
import re

def load_dataset(root_dir, dataset_id):
    pattern = r'Dataset(\d+)_(.*)'
    target_folder = ''
    for folder in os.listdir(root_dir):
        match = re.search(pattern, folder)
        if match:
            if int(match.group(1)) == dataset_id:
                target_folder = folder
        else:
            continue
    if target_folder == '':
        raise ValueError('root_dir 경로에 해당하는 dataset_id를 가지는 폴더가 존재하지 않습니다!')
    result_path = os.path.join(root_dir, target_folder)
    return result_path


def make_dataset(root_dir, dataset_id, dataset_name):
    dataset_name = f"Dataset{str(dataset_id).zfill(3)}_{dataset_name}"
    result_path = os.path.join(root_dir, dataset_name)
    for filename in os.listdir(root_dir):
        if filename.startswith(f"Dataset{str(dataset_id)}"):
            print("이미 해당하는 Dataset의 폴더가 존재합니다. dataset을 해당 폴더에 덮어씌웁니다.")
            return os.path.join(root_dir, filename)
    os.mkdir(result_path)
    return result_path

def check_generate_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder) 

def make_raw_detail_dataset(dataset_path):
    tr_path = os.path.join(dataset_path, 'imagesTr')
    ts_path = os.path.join(dataset_path, 'imagesTs')
    tr_seg_path = os.path.join(dataset_path, 'labelsTr')
    ts_seg_path = os.path.join(dataset_path, 'labelsTs')
    
    check_generate_folder(tr_path)
    check_generate_folder(ts_path)
    check_generate_folder(tr_seg_path)
    check_generate_folder(ts_seg_path)
       
    return tr_path, ts_path, tr_seg_path, ts_seg_path