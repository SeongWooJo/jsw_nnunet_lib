import pandas as pd
import sys
import numpy as np

my_lib_path = '/home/user/seong_test/jsw_lib'
sys.path.append(my_lib_path)

from jsw_nnunet_lib.util.data_analysis import parse_endoscope_pattern, normalize_text, check_duplicate, get_case, get_name



## adenoma, carcinoma json 파일을 읽어서, file | a_region | c_region | a_box | c_box
## 의 형태를 가지는 pandas dataframe을 내보내는 함수
## 각 region은 여러 개가 생길 수 있으므로, list of list로 구성되며, 각 element는 (x,y) points로 구성되 바로 draw_polygon에 사용할 수 있다.

def make_dataframe(adenoma_data, carcinoma_data):
    img_dict = {'id':[],'file':[]}
    seg_dict = {'id':[],'region':[],'box':[]}

    id_list = []
    img_file_list = []

    seg_id_list = []
    point_list = []
    box_list = []

    """
    'images' : [
        {'id': 이미지 고유 id, 'width' : 이미지 가로 사이즈, 'height' : 이미지 세로 사이즈, 'file_name' : '해당하는 실제 파일 이름'},
        ...
        ]
    'annotations' : [
        {'id' : annotation의 고유 id, 'iscrowd' : ??, 'image_id' : 해당 annotation이 그려지는 image_id, 'category_id' : carcinoma, adenoma 구분 추정
        , 'segmentation' : [[x1, y1, x2, y2, ... , x_n, y_n]]
        , 'bbox' : [x1, y1, width, height]
        , 'area' : 박스 area 
        },
        ...
        ]
    """

    ## adenoma json파일에서 images 키에 데이터를 가져옴
    images = adenoma_data.get('images', []) 
    
    ## adenoma json파일에서 annotations 키에 데이터를 가져옴
    annotations = adenoma_data.get('annotations', [])

    for instance in images:
        id = instance.get('id')
        file_path = instance.get('file_name')
        _, _, _, _, avilable = parse_endoscope_pattern(file_path)
        if not avilable:
            continue
        id_list.append(id)
        img_file_list.append(file_path)
    
    for instance in annotations:
        # json file annotation key 추출
        id = instance.get('image_id') # annotation에 해당하는 image id 추출
        seg_id_list.append(id)

        # annotation의 segmentation 추출
        segmentations = instance.get('segmentation', [])

        all_points_x = []
        all_points_y = []    
        for idx, point in enumerate(segmentations[0]):
            if idx % 2 == 0:
                all_points_x.append(point)
            else:
                all_points_y.append(point)
        
        if len(all_points_x) == len(all_points_y):
            points = list(zip(all_points_x, all_points_y))
            points.append(points[0])
        point_list.append(points)
        
        # annotation의 box 추출
        boxs = instance.get('bbox', [])
        box_list.append(boxs)

    img_dict['id'] = id_list
    img_dict['file'] = img_file_list

    seg_dict['id'] = seg_id_list
    seg_dict['region'] = point_list
    seg_dict['box'] = box_list


    a_image_df = pd.DataFrame(img_dict)    
    a_annotation_df = pd.DataFrame(seg_dict)

    id_list = []
    img_file_list = []

    seg_id_list = []
    point_list = []
    box_list = []

    images = carcinoma_data.get('images', [])
    annotations = carcinoma_data.get('annotations', [])


    for instance in images:
        id = instance.get('id')
        file_path = instance.get('file_name')
        _, _, _, _, avilable = parse_endoscope_pattern(file_path)
        if not avilable:
            continue
        id_list.append(id)
        img_file_list.append(file_path)

    for instance in annotations:
        # json file annotation key 추출
        id = instance.get('image_id')
        seg_id_list.append(id)

        # annotation의 segmentation 추출
        segmentations = instance.get('segmentation', [])

        all_points_x = []
        all_points_y = []    
        for idx, point in enumerate(segmentations[0]):
            if idx % 2 == 0:
                all_points_x.append(point)
            else:
                all_points_y.append(point)
        
        if len(all_points_x) == len(all_points_y):
            points = list(zip(all_points_x, all_points_y))
            points.append(points[0])
            
        point_list.append(points)

        # annotation의 box 추출
        boxs = instance.get('bbox', [])
        box_list.append(boxs)

    img_dict['id'] = id_list
    img_dict['file'] = img_file_list

    seg_dict['id'] = seg_id_list
    seg_dict['region'] = point_list
    seg_dict['box'] = box_list


    c_image_df = pd.DataFrame(img_dict)    
    c_annotation_df = pd.DataFrame(seg_dict)

    a_seg_df = pd.merge(a_image_df, a_annotation_df, left_on='id', right_on='id',how='inner')
    c_seg_df = pd.merge(c_image_df, c_annotation_df, left_on='id', right_on='id',how='inner')

    result_c_seg_df = c_seg_df.groupby('id').agg({
        'id' : 'first',
        'file': 'first',
        'region' : list,
        'box' : list,
    }).reset_index(drop=True)

    result_c_seg_df = result_c_seg_df.rename(columns={'region':'c_region','box':'c_box'})

    result_a_seg_df = a_seg_df.groupby('id').agg({
        'id' : 'first',
        'file': 'first',
        'region' : list,
        'box' : list,
    }).reset_index(drop=True)

    result_a_seg_df = result_a_seg_df.rename(columns={'region':'a_region','box':'a_box'})

    result_seg_df = pd.merge(result_c_seg_df, result_a_seg_df, left_on='file', right_on='file',how='outer')[['file','a_region','c_region','a_box','c_box']]
    result_seg_df['case'] = result_seg_df['file']
    result_seg_df['name'] = result_seg_df['file']

    result_seg_df['case'] = result_seg_df['case'].apply(get_case)
    result_seg_df['name'] = result_seg_df['name'].apply(get_name)

    return result_seg_df

def return_fold(fold_list, target):
    target_num = -1
    for idx, fold in enumerate(fold_list):
        if target in fold:
            target_num = idx
    
    return target_num

def seperate_train_test(df, key, fold_num=5, np_seed = 12345):
    try:
        unique_df = df[key].unique()
        np.random.seed(np_seed)
        np.random.shuffle(unique_df)
        fold_threshold = 1 / fold_num
        
        uni_num = len(unique_df)
        fold = []

        for i in range(fold_num):
            fold_lower_bound = int(i * fold_threshold * uni_num)
            fold_upper_bound = int((i+1) * fold_threshold * uni_num)
            if fold_upper_bound > uni_num:
                fold_upper_bound = uni_num
            
            fold.append(unique_df[fold_lower_bound:fold_upper_bound])
        
        for i in range(fold_num):
            df['set'] = df[key].apply(lambda x: f'fold_{return_fold(fold, x)}')

    except KeyError as e:
        print(f'Key Error 발생: {e}')
    
    return df