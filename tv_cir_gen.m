function  [cirmat,scf_mat,acf_mat] = tv_cir_gen(sr, cot_all, T) % exponential
%[cirmat]=tv_cir_gen(sr, cot) generates scatterring function with
%exponetial autocorrelation curve, given specified coherent time
%Inputs:
%   cot_all: coherence time in s, 0.8 as the threshold, can be a vector
%   sr: signal/cir sampling rate
%   T: the duration of the autocorrelation function(single side)
%Outputs:
%   cirmat
%Example:   [cirmat] = tv_cir_gen(5e3, 0.7, 1).
%By Zheng Guo, University of Alabama
%Created Feb 17, 2021.

%% ****** parameters ******%%

cht_threshold = 0.8; % this is the threshold that autocorrelation function falls to
N_tap = length(cot_all); % # of taps
t_vec = -T/2:1/sr:T/2-1/sr;
acf_mat = zeros(length(t_vec),N_tap);
scf_mat = zeros(length(t_vec),N_tap);
cirmat = zeros(length(t_vec),N_tap);
alpha = zeros(1, N_tap);

%% ****** generate tv cir for each tap ******%%
for i1 = 1:N_tap
    cot = cot_all(i1);
%%-&&&&& get autocorrelation function &&&&&-%%
    alpha_tmp = -log(cht_threshold)./cot; 
    alpha(i1) = alpha_tmp;
    acf = exp(-abs(alpha_tmp*t_vec));% this is the autocorrelation
    acf_mat(:,i1) = acf;
%%-&&&&& get scattering function &&&&&-%%
    nfft = length(acf);
    scf_mat(:,i1) = fft(acf.', nfft);
    
%%-&&&&& generate tv cir &&&&&-%%
    S_tmp = scf_mat(:,i1); % the filter for random g_tmp
    Nsim = size(scf_mat,1); % % ttl # of simulated cirs
    g_tmp = randn(Nsim,1)+1j*randn(Nsim,1);
    cir_tmp_fd = g_tmp.*sqrt(S_tmp); % Gaussian RVs shaped by S_tmp
    cir_tmp_td = ifft(cir_tmp_fd,Nsim); 
    cir_sim = cir_tmp_td;
    %%---- normalize to unit cir energy ----%%
    norm_factor = var(cir_sim,0,1);
    cir = cir_sim/sqrt(norm_factor);
    cirmat(:,i1) = cir;
    
    %figure;plot(acf_mat)
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%End of file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
