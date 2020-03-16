import re
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from infer_img import get_pred_img
from infer_vid import get_pred_vid

pred_img = get_pred_img()
pred_vid = get_pred_vid()

pred_img_df = pd.DataFrame.from_dict(pred_img)
pred_vid_df = pd.DataFrame.from_dict(pred_vid)

gallary = np.stack(pred_img_df['feat'])
query_grouped = pred_vid_df.groupby('item_id')

query_list = []
frame_list = []
retrieve_list = []
similarity_list = []

image_names = []
item_boxes = []
frame_boxes = []

for item_id, _df in tqdm(query_grouped):
    
    query_feats = np.stack(_df['feat'])
    sim_mat = query_feats.dot(gallary.T)
    retrieve_indeces = sim_mat.argmax(1)
    retrieve_item_ids = pred_img_df.loc[retrieve_indeces, 'item_id']
    retrieve_item_id = retrieve_item_ids.value_counts()[[0]]
    
    retrieved = retrieve_item_id.index[0]
    retrieve_item_id_idx = retrieve_item_ids.index[retrieve_item_ids == retrieved].unique().values
    
    query_idx = np.isin(retrieve_indeces, retrieve_item_id_idx)
    frame = _df.loc[_df['score'] == _df.iloc[query_idx]['score'].max(), 'frame'].iloc[0]
    frame_box = _df.loc[_df['score'] == _df.iloc[query_idx]['score'].max(), 'bbox'].iloc[0]
    retrieved_sim = sim_mat.max(1)[query_idx].mean()
    
    item_df = pred_img_df.loc[pred_img_df['item_id'] == retrieved]
    image_name = re.search('/image/{}/(.*).jpg'.format(retrieved), 
                       item_df.loc[item_df['score'] == item_df['score'].max(), 'file_name'].iloc[0]).group(1)
    item_box = item_df.loc[item_df['score'] == item_df['score'].max(), 'bbox'].iloc[0]
    
    query_list.append(item_id)
    frame_list.append(frame)
    retrieve_list.append(retrieved)
    similarity_list.append(retrieved_sim)
    
    image_names.append(image_name)
    item_boxes.append(item_box)
    frame_boxes.append(frame_box)


result_df = pd.DataFrame({'query': query_list, 
                          'item_id': retrieve_list,
                          'frame_index': frame_list,
                          'img_name': image_names,
                          'item_box': item_boxes,
                          'frame_box': frame_boxes,
                          
                          'similarity': similarity_list
                         })


final_dict = {}
for k, v in result_df.set_index('query').to_dict(orient='index').items():
    if v['similarity'] <= 0.39:
        continue

    final_dict[k] = {
        'item_id': v['item_id'],
        'frame_index': v['frame_index'],
        'result': [
            {
                'img_name': v['img_name'],
                'item_box': v['item_box'],
                'frame_box': v['frame_box']
            }
        ]
    }

with open('../result.json', 'w', encoding='utf-8') as f:
    json.dump(final_dict, f)