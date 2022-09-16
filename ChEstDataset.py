import torch
import os
import scipy.io


class ChEstDataset(torch.utils.data.Dataset):
    # Dataset for Channel Estimation Simulation
    def __init__(self, root_dir, SNR, test, blk_sample, sr, sym_blk):
        self.mat_files = list()
        self.blk_sample = blk_sample
        self.sr = sr
        self.sym_blk = sym_blk
        for file in os.listdir(root_dir):
            if test:
                if file.startswith("tv_simple_test_" + str(SNR)):
                    self.mat_files.append(os.path.join(root_dir, file))
            else:
                if file.startswith("tv_simple_" + str(SNR)):
                    self.mat_files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.mat_files) * ((self.sr // self.sym_blk) // self.blk_sample)

    def __getitem__(self, idx):
        # Get working with csv/non-MAT file
        file_num = idx // ((self.sr // self.sym_blk) // self.blk_sample)
        blk_num = idx % ((self.sr // self.sym_blk) // self.blk_sample)
        mat_file = scipy.io.loadmat(self.mat_files[file_num])
        tx = mat_file['tx_symbols']
        y = mat_file['y']
        cirmat_ls = mat_file['cirmat_ls']
        cirmat = mat_file['cirmat']
        sample = {'usb': tx[blk_num * self.blk_sample * self.sym_blk:(blk_num + 1) * self.blk_sample * self.sym_blk],
                  'r': y[blk_num * self.blk_sample * self.sym_blk:(blk_num + 1) * self.blk_sample * self.sym_blk],
                  'h_ls': cirmat_ls[blk_num * self.blk_sample:(blk_num + 1) * self.blk_sample, :],
                  'h': cirmat[blk_num * self.blk_sample * self.sym_blk:(blk_num + 1) * self.blk_sample * self.sym_blk,
                       :]}
        return sample
