clear variables
close all
rng('default');
global flag

files = dir('./test_data');
lists = get_list(files);
for i=1:length(lists)
    clearvars new
    flag = false;
    data = csvread(lists(i));
%     data = gen_data([1,1],[1,0;0,1],30, [3,3],[2,1;1,2],100);
    x = data(:, 1:end-1);
    y = data(:, end);
%     plot_fig(x, y)
    new = find_overlapped(x, y);
%     plot_fig(new.x, new.y)
    csvwrite(lists(i), [new.x, new.y]);
    
    c = cvpartition(new.y,'KFold',3);
%     train_x = new.x(c.training(2), :); train_y = full(ind2vec(new.y(c.training(2))'+1))';
%     test_x = new.x(c.test(2), :); test_y = new.y(c.test(2));
%     node = mvid3fast(train_x, train_y);
%     pred_y = uclass(node, test_x);
%     [~, idx] = max(pred_y, [], 2);
%     pred_y = idx - 1;
    Mdl = fitcsvm(new.x(c.training(2), :), new.y(c.training(2), :), 'ClassNames', [1 0], 'KernelFunction','rbf');
    [pred_y,score,cost] = predict(Mdl,new.x(c.test(2), :));
    disp(lists(i))
%     cp = classperf(new.y(c.test(2), :), pred_y);
%     disp(cp.CorrectRate)
    disp(strcat('POS: ', string(sum(new.y==1)), '/', string(sum(y==1))))
    disp(strcat('NEG: ', string(sum(new.y==0)), '/', string(sum(y==0))))
    confmat(new.y(c.test(2), :), pred_y)
end

%%
function [] = plot_fig(x, y)
    d1 = sum(y==1);
    
    figure
    hold on
    scatter(x(1:d1,1),x(1:d1,2), '+', 'b')
    scatter(x(d1+1:end,1),x(d1+1:end,2), '.', 'r')
end

%%
function new = find_overlapped(x, y)
    idx.one = find(y == 1);
    idx.zero = find(y == 0);
    
    sample = prdataset(x, y);
    one = target_class(sample, '1');
    zero = target_class(sample, '0');
    mapped = mapping_warpper(one, zero);
    
    class_overlapped = get_idx(mapped);
    overlapped.one = idx.one(class_overlapped.one);
    overlapped.zero = idx.zero(class_overlapped.zero);
    overlapped = [overlapped.one; overlapped.zero];
    
    idx = ones(size(y)); idx(overlapped) = 0;
    new.x = x(logical(idx), :);
    new.y = y(logical(idx));
    return
end

%%
function blob = gen_data(m1, s1, d1, m2, s2, d2)
    x1 = mvnrnd(m1, s1, d1); y1=ones(d1,1);
    x2 = mvnrnd(m2, s2, d2); y2=zeros(d2,1);
    data1 = [x1, y1];
    data2 = [x2, y2];
    blob = [data1; data2];
    return
end

%%
function mapped = mapping_warpper(one, zero)
    mapped.zero = mapping(one, zero);
    mapped.one = mapping(zero, one);
    return
end

%%
function out = mapping(base, bymapped)
    global flag
    w = svdd(base);
    if flag
        plotc(w)
    end
    out = bymapped*w;
    out = out.data;
    return
end

%%
function overlapped_idx = get_idx(mapped)
    overlapped_idx.one = find(comp(mapped.one)>0);
    overlapped_idx.zero = find(comp(mapped.zero)>0);
    return
end

%%
function idx = comp(array)
    idx = abs(array(:, 1)) < abs(array(:, 2));
    return
end

%%
function [lists] = get_list(files)
    lists = []; 
    [size_, ~] = size(files);
    for i=1:size_
        if files(i).isdir
            continue
        end
        lists = [lists, string(files(i).name)];
    end
end
