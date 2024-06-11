import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("result_file", help="/media/renhao/Research/GitHub/trajectory_prediction/SIT/results_sit.pkl")
parser.add_argument("--vis", type = str, default="None")
parser.add_argument("--rw_input", action="store_true")
args = parser.parse_args()

with open(args.result_file, "rb") as f:
    data = pickle.load(f)
    
SCALE_MAP = {
            "0000": 26,
            "0500": 25,
            "0401": 20,
            "0400": 16,
            "eth": 37.5,
            "hotel": 135,
            "zara01": 70,

        }

def metrics(pred, all_futures):
    """
    pred: [K, 12, 2]
    all_futures: list of [12,2]
    """
    
    all_hits = []
    all_distance = []
    for gt_traj in all_futures:
        gt_traj_12 = gt_traj[None, :pred.shape[1]]
        valid_len = gt_traj_12.shape[1]
        distance = np.linalg.norm(pred[:,:valid_len] - gt_traj_12 , axis=-1)
        all_distance.append(distance)
    ade = min([i.mean(-1).min() for i in all_distance])
    fde = min([i[...,-1].min() for i in all_distance])
    recall = np.less(np.stack([i.mean(-1) for i in all_distance], axis=0), 2).any(0).mean()
    precision =  np.less(np.stack([i.mean(-1) for i in all_distance], axis=0), 2).any(1).mean()
    usage = np.unique(np.stack([i.mean(-1) for i in all_distance], axis=0).argmin(0)).shape[0] / len(all_futures)
    return dict(ade = ade, fde = fde, recall = recall, precision = precision, ptu = usage)

results_dict = dict(
    name = [],
    ade = [],
    fde = [],
    recall =[],
    precision = [],
    ptu = [],
)
    
for scene, pred_data in data.items():
    with open(f"../next_x_v1_dataset_prepared_data/multifuture/test/{scene}.p", "rb") as f:
        multifuture = pickle.load(f)
        
    all_futures = []
    for future_id in multifuture:
        gt_traj = multifuture[future_id]["x_agent_traj"]
        gt_traj = np.array([one[2:] for one in gt_traj])
        all_futures.append(gt_traj)
        obs_traj = np.array(multifuture[future_id]['obs_traj'])
        target_id = future_id.split('_')[2]
    
    scale = SCALE_MAP[scene.split('_')[0]]
    all_futures = [gt / scale for gt in all_futures]
    for  pred, trackId in zip( pred_data['pred'], pred_data['trackId']):
        if not args.rw_input: pred = pred / scale
        if str(trackId) == target_id:
            # for i in                 import pdb; pdb.set_trace()range(20):
            #     plt.plot(pred[0,i,:,0],pred[0,i,:,1], "*")
            if scene.split("_")[0] == args.vis:
                for gt_traj in all_futures:
                    plt.plot(gt_traj[:12,0] * scale, gt_traj[:12,1] * scale, "b*")
                plt.rcParams['figure.figsize'] = (20,10)
                plt.imshow(cv2.imread(f"../next_x_v1_dataset_prepared_data/seg/{scene}_id.jpg",0))
                plt.plot(obs_traj[:, 2], obs_traj[:,3], "r*-")
                # plt.plot(obs[:, 0], obs[:,1], "ro")
                plt.title(scene)

                for i in range(20):
                    plt.plot(pred[0,i,:,0] * scale,pred[0,i,:,1] * scale, "g*-", alpha=0.7)
                plt.show()
            try:
                for k, v in metrics(pred[0], all_futures).items():
                    results_dict[k].append(v)
                results_dict['name'].append(scene.split("_")[0])
                
                # results_dict for k, v in metrics(pred[0], all_futures).items
            except:
                raise
        
            break


    
# all_ade = {k: sum(v)/len(v) for k, v in all_ade.items()}
# all_fde = {k: sum(v)/len(v) for k, v in all_fde.items()}
# all_precision = {k: sum(v)/len(v) for k, v in all_precision.items()}
# all_usage = {k: sum(v)/len(v) for k, v in all_usage.items()}

result_table = pd.DataFrame(results_dict)
# print(result_table)
print(len(result_table), "in total")
# print(result_table.groupby("name").mean() )
# print(result_table.groupby("name").count())

for k, v in result_table.mean().items():
    if k in ['recall', 'precision', 'ptu']:
        print(f'& {v * 100:.2f}\%', end = " ")
    else:
        print(f'& {v:.2f}', end = " ")
print("")
# for scene_group in [["0000","0400","0401", "0500", "eth", "hotel", "zara10"],['zara01']]:
#     virat_df = result_table[result_table.name.isin(scene_group)].loc[:,["total", "ade", "fde", "usage"]]
#     ade = (virat_df.ade * virat_df.total).sum() / virat_df['total'].sum()
#     fde = (virat_df.fde * virat_df.total).sum() / virat_df['total'].sum()
#     usage = (virat_df.usage * virat_df.total).sum() / virat_df['total'].sum()
#     print(scene_group, "ADE:", "%.2f"%ade, "FDE", "%.2f"%fde, "Usage", "%.2f"%(usage * 100) + "%")

# for scene_group in [["0000","0400","0401", "0500", "eth", "hotel", "zara10"],['zara01']]:
#     virat_df = result_table[result_table.name.isin(scene_group)].loc[:,["total", "ade", "fde", "usage"]]
#     ade = (virat_df.ade * virat_df.total).sum() / virat_df['total'].sum()
#     fde = (virat_df.fde * virat_df.total).sum() / virat_df['total'].sum()
#     usage = (virat_df.usage * virat_df.total).sum() / virat_df['total'].sum()
#     print( "%.2f"%ade, "&", "%.2f"%fde, "&", "%.2f"%(usage * 100) + "\%", end = "&" )