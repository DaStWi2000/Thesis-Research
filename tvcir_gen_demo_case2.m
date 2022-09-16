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
channel_array=cot_all; % rename cot_all to more reasonable name
h_keep = h;
channel_array_keep = channel_array;

num_files = 1;
for file_num = 1:num_files
    h = h_keep;
    channel_array = channel_array_keep;

    %% ****** tv channel generation ******%%
    % case 1
    % tv_path_idx = [4,5]; % the tap that is time-varying
    % channel_array = [0.036,0.05]; % coherent time

    % case 2
    % tv_path_idx_tmp = find(h~=0);
    % tv_path_idx = tv_path_idx_tmp(2:14);% the tap that is time-varying
    % channel_array = channel_array(2:14); % first tap assumed static
    h_indexes_temp = find(h~=0);
    %find acquires nonzero h
    h = h(h_indexes_temp(1:2));
    %set h to h without zero terms, only the first 2 items (1:2)
    h_indexes = [2];
    channel_array = channel_array(2);
    
    channel_order = length(h);% channel order
    step_rate = 5e3; % symbol rate
    duration = 1; % in s
    snr = 20; % in dB
    step_number = round(step_rate*duration); %data length
    block_length = 3*channel_order;% observation window size

    binary_sequence=randi(2, [step_number, 1])-1;
    tx_symbols=2*binary_sequence-1;           %Mapping
    
    [time_varying_temp,~,~] = tv_cir_gen(step_rate, channel_array,duration);
    channel_scale_factors_temp  = repmat(h.',step_number,1);
    channel_scales = channel_scale_factors_temp;

    for ix = 1:length(h_indexes)
        channel_scales(:,h_indexes(ix)) =  h(h_indexes(ix))*time_varying_temp(:,ix);
    end

    %------ get received signal ------%
    N_column = size(channel_scales,2)+length(tx_symbols)-1;
    convolution_matrix = zeros(length(tx_symbols),N_column); % channel convolution matrix
    for i5 = 1:length(tx_symbols)
        h_tr_tmp = channel_scales(i5,:);
        h_instant = h_tr_tmp(end:-1:1);   % current CIR
        pad_pre_len = i5-1;
        pad_post_len = N_column-pad_pre_len;
        convolution_column = [zeros(1,pad_pre_len),h_instant,...
            zeros(1,pad_post_len-length(h_instant))];
        convolution_matrix(i5,:) = convolution_column;
    end
    insignificant_digits = zeros(length(h)-1,1);
    signal_vector = [insignificant_digits;tx_symbols];  % length(tx_pre)=cir_length-1;
    result_noiseless = convolution_matrix*signal_vector; % noise free data

    %------ add noise ------%
    volume_scale = 10^(-snr/10); % in (w)
    noise = sqrt(volume_scale/2)*(randn(1,step_number)+1i*randn(1,step_number));% noise
    result_noisy = result_noiseless+noise.'; % noisy

    %% ****** Channel estimation ******%%
    blk_n = step_number;%floor(step_number/block_length);
    cirmat_ls = zeros(blk_n,channel_order);
    for blk_idx = 1:blk_n
        for sym_idx = 1:block_length
            x = (blk_idx-1)+sym_idx+channel_order-1;
            y=(blk_idx-1)+sym_idx;
            a=5;
            %if blk_idx==5000
            %    a=5;
            %end
            if max(x,y)>step_number+1
                continue
            end
            %[sym_idx,blk_idx]
            A_mtx(sym_idx,:) = signal_vector( ...
                (blk_idx-1)*1+sym_idx+channel_order-1: ...
                -1: ...
                (blk_idx-1)*1+sym_idx);
            %A_mtx(sym_idx,:) = signal_vector( ...
            %    (blk_idx-1)*block_length+sym_idx+channel_order-1: ...
            %    -1: ...
            %    (blk_idx-1)*block_length+sym_idx);
        end
        if blk_idx>step_number-block_length+1
            x=zeros(1,blk_idx+block_length-step_number-1);
            y=result_noisy(blk_idx:step_number)';
            size(x);
            size(y);
            rx_blk_symbols = [y x].';
        else
            rx_blk_symbols = result_noisy(blk_idx:blk_idx+block_length-1);
        end
        %rx_blk_symbols = result_noisy((blk_idx-1)*block_length+1:blk_idx*block_length);
        %------ LS channel estimation ------%
        p_LS = (A_mtx'*A_mtx)\(A_mtx'); % LS matrix
        if p_LS ~= p_LS
            h_est_LS = cirmat_ls(blk_idx-1,:);
        else
            h_est_LS =  (p_LS*rx_blk_symbols).';% LS estimate
        end
        cirmat_ls(blk_idx,:) = h_est_LS;
    end
    save(['Dataset\\tv_simple_test_',num2str(snr),'_',num2str(file_num),'.mat'],'tx_symbols','channel_scales','cirmat_ls','result_noisy')
end
plot(real(cirmat_ls(:,:)))
figure
plot(real(channel_scales(:,:)))