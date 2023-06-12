#improts
import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import numpy as np

#function to create dataset dictionary with input and grpund truth
def create_dict(folder, NET, NET_idx, H):
    Data_dict = {}
    #opening pickle
    with open(osp.join(folder, '{}_{}_{}.pkl'.format(H, NET,NET_idx)), 'rb') as file:
            data_vis = pickle.load(file)
    #calc num voxels for transfromer head size
    num_voxels = data_vis.shape[1]

    for i in range(15): #number of classes = 15
        #extracting movie data from all data
        temp_data = temp_data.loc[data_vis['y'] == i]

        # padding each movie to be the length of the longest movie (336)
        padd_size = 336 - temp_data.shape[0]
        if padd_size != 0:
            data_shape = temp_data.shape[1]
            padding_data = np.zeros((padd_size, data_shape))
            #setting the values of the clip & subjects to be consistant after padding
            padding_data[:,data_shape - 2], padding_data[:,data_shape - 1] = i, temp_data["Subject"][temp_data["Subject"].index[0]]
            padding_df_vis = pd.DataFrame(padding_data, columns=temp_data.columns)
            temp_data = pd.concat([temp_data, padding_df_vis], ignore_index=True)
        if temp_data.shape[0] != 336:
            raise Exception('not padded, shape is ', temp_data.shape[0], 'at ', '{}_{}_{}.pkl'.format(H, NET,NET_idx))

        temp_data.astype({"timepoint": int, "y": int, "Subject": int})
        #removing first and last 5 TRs in case there was a delay
        ROI_net = temp_data.iloc[4:temp_data.shape[0] - 5, 0:temp_data.shape[1] - 3]
        ROI_net = ROI_net.reset_index().set_index('index')
        Clip = i
        Subject = temp_data['Subject'][temp_data['Subject'].index[0]]


        net_values = ROI_net.copy()
        #info string containing the subject number, net info and movie index
        info_string = 'input_data_{}_{}_{}_{}'.format(NET, H, Clip, round(Subject))
        #grpund truth one-hot encoded
        Clip_gt = [0.0 for i in range(15)]
        Clip_gt[Clip] = 1.0
        #adding this sample to data dictionary
        #shape of data is a tesnor of voxels x time
        Data_dict[info_string] = {"vis_values": torch.tensor(net_values.values, dtype=torch.float32),
                                  "clip_idx": torch.tensor(Clip_gt),
                              "columns": list(net_values.columns)}


    return Data_dict, num_voxels

def create_dataset(directory, phase, net,NET_idx, hem):
    data_dict = {}
    for subject_folder in glob.iglob(directory + '/' + phase + '/' +  '/**/'):
        subject_dict, num_voxels = create_dict(subject_folder, net, NET_idx, hem)
        data_dict.update(subject_dict)
    return data_dict, num_voxels-3 # -3 columns of clip subject and timepoint
