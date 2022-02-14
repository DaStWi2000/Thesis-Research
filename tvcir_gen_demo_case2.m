%This program simulates time-varying channel based reception and performs
%   LS channel estimation in baseband
%Created on Mar 2,2021
%By Zheng Joshua Guo, University of Alabama
%Updated on Mar 9,2021.

clear all; clc; close all;
%% ****** parameters ******%%
% case 1
% h= [0.04 -0.05 0.07 -0.21 -0.5 0.72 0.36 0 0.21 0.03 0.07].'; %channel A
% case 2
load('exp_cir.mat','h','cot_all')  % channel from 2016Gulf
h_keep = h;
cot_all_keep = cot_all;

num_files = 500;
for file_num = 1:num_files
    h = h_keep;
    cot_all = cot_all_keep;
    cir_length = length(h);% channel order
    sr = 5e3; % symbol rate
    duration = 1; % in s
    snr = 20; % in dB
    N_sym = round(sr*duration); %data length
    blk_len = 2*cir_length;% observation window size

    vInfo=randi(2, [N_sym, 1])-1;
    tx_symbols=2*vInfo-1;           %Mapping

    %% ****** tv channel generation ******%%
    % case 1
    % tv_path_idx = [4,5]; % the tap that is time-varying
    % cot_all = [0.036,0.05]; % coherent time

    % case 2
    tv_path_idx_tmp = find(h~=0);
    tv_path_idx = tv_path_idx_tmp(2:14);% the tap that is time-varying
    cot_all = cot_all(2:14); % first tap assumed static

    [tvcir_tmp,~,~] = tv_cir_gen(sr, cot_all,duration);
    cir_mat_tmp  = repmat(h.',N_sym,1);
    cirmat = cir_mat_tmp;

    for it_idx = 1:length(tv_path_idx)
        cirmat(:,tv_path_idx(it_idx)) =  h(tv_path_idx(it_idx))*tvcir_tmp(:,it_idx);
    end

    %------ get received signal ------%
    N_column = size(cirmat,2)+length(tx_symbols)-1;
    H_mtx_iv_loop = zeros(length(tx_symbols),N_column); % channel convolution matrix
    for i5 = 1:length(tx_symbols)
        h_tr_tmp = cirmat(i5,:);
        h_instant = h_tr_tmp(end:-1:1);   % current CIR
        pad_pre_len = i5-1;
        pad_post_len = N_column-pad_pre_len;
        H_mtx_iv_loop_tmp = [zeros(1,pad_pre_len),h_instant,...
            zeros(1,pad_post_len-length(h_instant))];
        H_mtx_iv_loop(i5,:) = H_mtx_iv_loop_tmp;
    end
    tx_pre = zeros(length(h)-1,1);
    tx_pre_pad = [tx_pre;tx_symbols];  % length(tx_pre)=cir_length-1;
    rx = H_mtx_iv_loop*tx_pre_pad; % noise free data

    %------ add noise ------%
    N0 = 10^(-snr/10); % in (w)
    noise = sqrt(N0/2)*(randn(1,N_sym)+1i*randn(1,N_sym));% noise
    y = rx+noise.'; % noisy

    %% ****** Channel estimation ******%%
    blk_n = floor(N_sym/blk_len);
    cirmat_ls = zeros(blk_n,cir_length);
    for blk_idx = 1:blk_n
        for sym_idx = 1:blk_len
            A_mtx(sym_idx,:) = tx_pre_pad((blk_idx-1)*blk_len+sym_idx+cir_length-1:-1:(blk_idx-1)*blk_len+sym_idx);
        end
        rx_blk_symbols = y((blk_idx-1)*blk_len+1:blk_idx*blk_len);
        %------ LS channel estimation ------%
        p_LS = (A_mtx'*A_mtx)\(A_mtx'); % LS matrix
        h_est_LS =  (p_LS*rx_blk_symbols).';% LS estimate
        cirmat_ls(blk_idx,:) = h_est_LS;
    end
    save(['Dataset\\tv_',num2str(snr),'_',num2str(file_num),'.mat'],'tx_symbols','cirmat','cirmat_ls','y')
end