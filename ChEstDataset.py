import torch
import os
import scipy.io

class ChEstDataset(torch.utils.data.Dataset):
    # Dataset for Channel Estimation Simulation
    def __init__(self, root_dir, SNR, test):
        self.mat_files = list()
        for file in os.listdir(root_dir):
            if test == True:
                if file.startswith("tv_test_"+str(SNR)):
                    self.mat_files.append(os.path.join(root_dir,file))
            else:
                if file.startswith("tv_"+str(SNR)):
                    self.mat_files.append(os.path.join(root_dir,file))

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # Get working with csv/non-MAT file
        mat_file = scipy.io.loadmat(self.mat_files[idx])
        tx = mat_file['tx_symbols']
        y = mat_file['y']
        cirmat_ls = mat_file['cirmat_ls']
        cirmat = mat_file['cirmat']
        sample = {'usb' : tx, 'r' : y, 'h_ls' : cirmat_ls, 'h' : cirmat}
        return sample
