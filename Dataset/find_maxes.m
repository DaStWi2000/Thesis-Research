close all; clear all; clc;
max_cirmat_r = -Inf
min_cirmat_r = Inf
max_ls_r = -Inf
min_ls_r = Inf
max_y_r = -Inf
min_y_r = Inf
max_cirmat_i = -Inf
min_cirmat_i = Inf
max_ls_i = -Inf
min_ls_i = Inf
max_y_i = -Inf
min_y_i = Inf
file_prefix = 'tv_simple_test_20_'
for i = 1:75
    file = open([file_prefix,num2str(i),'.mat']);
    cirmat = file.cirmat;
    cirmat_ls = file.cirmat_ls;
    y = file.y;
    if max_cirmat_r < max(real(cirmat),[],'all')
        max_cirmat_r = max(real(cirmat),[],'all');
    end
    if min_cirmat_r > min(real(cirmat),[],'all')
        min_cirmat_r = min(real(cirmat),[],'all');
    end
    if max_ls_r < max(real(cirmat_ls),[],'all')
        max_ls_r = max(real(cirmat_ls),[],'all');
    end
    if min_ls_r > min(real(cirmat_ls),[],'all')
        min_ls_r = min(real(cirmat_ls),[],'all');
    end
    if max_y_r < max(real(y),[],'all')
        max_y_r = max(real(y),[],'all');
    end
    if min_y_r > min(real(y),[],'all')
        min_y_r = min(real(y),[],'all');
    end
    if max_cirmat_i < max(imag(cirmat),[],'all')
        max_cirmat_i = max(imag(cirmat),[],'all');
    end
    if min_cirmat_i > min(imag(cirmat),[],'all')
        min_cirmat_i = min(imag(cirmat),[],'all');
    end
    if max_ls_i < max(imag(cirmat_ls),[],'all')
        max_ls_i = max(imag(cirmat_ls),[],'all');
    end
    if min_ls_i > min(imag(cirmat_ls),[],'all')
        min_ls_i = min(imag(cirmat_ls),[],'all');
    end
    if max_y_i < max(imag(y),[],'all')
        max_y_i = max(imag(y),[],'all');
    end
    if min_y_i > min(imag(y),[],'all')
        min_y_i = min(imag(y),[],'all');
    end
    i
end
clearvars cirmat cirmat_ls file i y

save([file_prefix,'summary.mat'])