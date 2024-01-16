import torch

class Dataset(torch.utils.data.Dataset):
    '''
    Dataloader to train a coarser (527 --> 3)
    '''
    def __init__(self, scores, gt):
        self.scores = scores
        self.gt = gt
        self.len_data = scores.shape[0]
    def __getitem__(self, idx):

        x = torch.tensor(self.scores[idx])
        y = torch.tensor(self.gt[idx])

        return (x, y)

    def __len__(self):
        return self.len_data








#INTERESTING IDEA: keep in mind if need to implement open multiple small files
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, f_fold, f_len, setting, experiment, batch_size):
#         # NOTE: SHUFFLE must be done manually, and not in dataloader, otherwise this will cause an issue
#         #f_folds contains the names of the files to load
#         #f_size contains the length of the files to load
#         # n_to_delete = -4

#         # for file in f_fold:
#         #     groundtruth_name = setting.replace('step', 'compute_groundtruth').replace('deep', 'False').identifier()+'_groundtruth.npy'
#         #     fname_name = setting.replace('step', 'compute_groundtruth').replace('deep', 'False').identifier()+'_fname.npy'
#         #     grountruth = np.load(experiment.path.groundtruth+groundtruth_name)
#         #     fname =  np.load(experiment.path.groundtruth+fname)

#         # self.data = scores[0,:,:]
#         self.total_len = f_len.sum()
#         self.f_fold = f_fold
#         self.idx_file = 0
#         self.cur_file = np.load(self.f_fold[0])
#         self.idx_f_len = 0
#     def __getitem__(self, idx):
#         try:
#             score_frame = self.cur_file[idx] 
#         except IndexError:
#             self.idx_file += 1
#             self.cur_file = np.load(self.f_fold)
#             score_frame = self.cur_file[idx] 

#         score_frame = torch.from_numpy(np.copy(score_frame))
#         return (score_frame)

#     def __len__(self):
#         return self.total_len