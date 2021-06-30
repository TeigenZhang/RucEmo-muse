import os
import h5py
import copy
import numpy as np

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .base_dataset import BaseDataset


class MusePhysioSlideDataset(BaseDataset):    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--win_len', type=int, default=300, help='window length of a segment')
        parser.add_argument('--hop_len', default=50, type=int, help='step length of a segmentt')
        
        return parser

    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst]
        '''
        super().__init__(opt)
        self.root = '/data12/lrc/MUSE2021/h5_data/c4_muse_physio/'
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.set_name = set_name
        self.load_label()
        self.load_feature()
        self.win_len = opt.win_len
        self.hop_len = opt.hop_len
        if set_name == 'trn':
            self.feature_segments, self.label_segments = self.make_segments()
        
        self.manual_collate_fn = False
        print(f"MuseWild dataset {set_name} created with total length: {len(self)}")
    
    def pad_max_len(self, tensor, max_len):
        # tensor = torch.from_numpy(tensor)
        # tensor -> T*D
        if len(tensor) < max_len:
            if tensor.ndim == 1:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor))], dim=0)
            else:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor), tensor.size(1))], dim=0)
        return tensor
    
    def make_segments(self):
        all_ft_segments = []
        all_label_segments = []
        for key in self.feature_data[list(self.feature_data.keys())[0]].keys():  #get the data order
            label_dict = self.target[key]
            feature = []
            for feature_name in self.feature_data.keys(): # concat feature
                feature.append(self.feature_data[feature_name][key])
            feature = torch.cat(feature, dim=-1)
            length = label_dict['length']
            stop_flag = False
            for st in range(0, length, self.hop_len):
                ed = st + self.win_len
                if ed > length:
                    ed = length
                    stop_flag = True

                label_seg = {
                    'EDA': self.pad_max_len(label_dict['EDA'][st:ed], self.win_len),
                    'length': torch.as_tensor(ed - st).long(),
                    'timestamp': self.pad_max_len(label_dict['timestamp'][st:ed], self.win_len)
                }
                all_label_segments.append(label_seg)
                ft_seg = self.pad_max_len(feature[st: ed, :], self.win_len)
                all_ft_segments.append(ft_seg)
                if stop_flag:
                    break
        return all_ft_segments, all_label_segments

    def load_label(self):
        partition_h5f = h5py.File(os.path.join(self.root, 'target', 'partition.h5'), 'r')
        self.seg_ids = sorted(partition_h5f[self.set_name])
        self.seg_ids = list(map(lambda x: str(x), self.seg_ids))
        label_h5f = h5py.File(os.path.join(self.root, 'target', '{}_target.h5'.format(self.set_name)), 'r')
        self.target = {}
        for _id in self.seg_ids:
            if self.set_name != 'tst':
                self.target[_id] = {
                    'EDA': torch.from_numpy(label_h5f[_id]['anno12_EDA'][()]).float(),
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }
            else:
                self.target[_id] = {
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }

    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
            feature_data = {}
            for _id in self.seg_ids:
                feature_data[_id] = torch.from_numpy(h5f[self.set_name][_id]['feature'][()])
                # assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                assert len(h5f[self.set_name][_id]['timestamp'][()]) == len(self.target[_id]['timestamp']), '\
                    Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
            self.feature_data[feature_name] = feature_data

    def __getitem__(self, index):
        if self.set_name == 'trn':
            ft_seg = self.feature_segments[index].float()
            label_seg = self.label_segments[index]
            length = self.label_segments[index]['length'].item()
            mask = torch.zeros(len(ft_seg)).float()
            mask[:length] = 1.0

            ## {feature:.., arousal:..., valence:..., timestamp:..., lenth}
            return {
                **{'feature': ft_seg, 'mask': mask}, **label_seg, **{"feature_names": self.feature_set}
            }
        else:
            _id = self.seg_ids[index]
            feature = torch.cat([
                self.feature_data[feat_name][_id] for feat_name in self.feature_set
            ], dim=-1)
            return {
                'feature': feature.float(),
                'EDA': self.target[_id]['EDA'].float(),
                'timestamp': self.target[_id]['timestamp'].float(),
                'mask': torch.ones(feature.size(0)).long()
            }

    def __len__(self):
        return len(self.feature_segments) if self.set_name == 'trn' else len(self.seg_ids)
    

if __name__ == '__main__':
    class test:
        feature_set = 'bert,vggface,vggish'
        dataroot = '/data12/lrc/MUSE2021/h5_data/c4_muse_stress/'
        win_len = 300
        hop_len = 100
        max_seq_len = 100

    opt = test()
    a = MusePhysioSlideDataset(opt, 'val')

    data1 = a[0]
    for key, value in data1.items():
        print(key, value)
    


    '''
    [t1, t2, t3, t4] t1 -> (D, )
    t4 -> zero_tensor
    [1, 1, 1, 0]
    [pred1, pred2, pred3, pred4] = [pred1, pred2, pred3, pred4] * [1, 1, 1, 0]
    [tgt, tgt, tgt, tgt] = [tgt, tgt, tgt, tgt] * [1, 1, 1, 0]
    MSE(P, T)
    '''

