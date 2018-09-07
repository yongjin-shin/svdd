clear variables
close all
rng('default');
global is_real_data

files = dir('./test_data');
lists = get_list(files);
for i=1:length(lists)
    disp(lists(i))
    clearvars new
    is_real_data = true;
%     data = gen_data([1,1],[1,0;0,1],30, [3,3],[2,1;1,2],100);
    data = csvread(lists(i));
    x = data(:, 1:end-1);
    y = data(:, end);
    c = cvpartition(y,'KFold',3);
    for j=1:c.NumTestSets   
        train_x = x(c.training(j), :); train_y = y(c.training(j));
        test_x = x(c.test(j), :); test_y = y(c.test(j));
        
        % -- PCA for plotting: dim=2 -- %
        [train_x, pca_mapping] = pca(train_x, 2);
        test_x = test_x * pca_mapping.M;
        plot_fig(train_x, train_y, test_x, test_y)
        
        % -- mapping and training classifier for train set -- %
        [train_idx, mapping] = find_overlapped(train_x, train_y);
        new_train.x = train_x(train_idx, :);
        new_train.y = train_y(train_idx);
%         Mdl = fitcsvm(new_train.x, new_train.y, 'ClassNames', [1 0], 'KernelFunction','rbf');
        new_train_y = full(ind2vec(new_train.y'+1))';
        node = mvid3fast(new_train.x, new_train_y);
        
        % -- mapping test set and classify -- %
        [test_dx, ~] = find_overlapped(test_x, test_y, mapping);
        new_test.x = test_x(test_dx, :);
        new_test.y = test_y(test_dx);
        plot_fig(new_train.x, new_train.y, new_test.x, new_test.y)
%         [pred_y,score,cost] = predict(Mdl, new_test.x);
        pred_y = uclass(node, new_test.x);
        [~, idx] = max(pred_y, [], 2);
        pred_y = idx - 1;
        
        % -- compare true and prediction -- %        
        confmat(new_test.y, pred_y)
        fprintf('Train-POS: %d(remained)/%d \t Train-Neg: %d(remained)/%d\n', sum(new_train.y==1), sum(train_y==1), sum(new_train.y==0), sum(train_y==0))
        fprintf('Train-POS: %d(remained)/%d \t Train-Neg: %d(remained)/%d\n', sum(new_test.y==1), sum(test_y==1), sum(new_test.y==0), sum(test_y==0))
%         cp = classperf(new_test.y, pred_y);
%         fprintf('Test Acc: %-.3f\n', (cp.CorrectRate));
    end
    fprintf('\n')
end

%%
function [new_idx, mapping] = find_overlapped(x, y, given_mapping)
    idx.one = find(y == 1);
    idx.zero = find(y == 0);
    
    sample = prdataset(x, y);
    one = target_class(sample, '1');
    zero = target_class(sample, '0');

    if nargin == 2
        mapping = mapping_warpper(one, zero);
    elseif nargin > 2
         mapping = given_mapping;
    end
    
    mapped.zero = mapping_result_of(zero, mapping.one);
    mapped.one = mapping_result_of(one, mapping.zero);
    
    new_idx = overlapped_region(idx, mapped, y);
    return
end

%%
function new = overlapped_region(idx, mapped, y)
    function overlapped_idx = get_idx(mapped)
        function idx = comp(array)
            idx = abs(array(:, 1)) < abs(array(:, 2));
            return
        end
        
        overlapped_idx.one = find(comp(mapped.one)>0);
        overlapped_idx.zero = find(comp(mapped.zero)>0);
        return
    end

    class_overlapped = get_idx(mapped);
    overlapped.one = idx.one(class_overlapped.one);
    overlapped.zero = idx.zero(class_overlapped.zero);
    overlapped = [overlapped.one; overlapped.zero];
    
    new = ones(size(y)); new(overlapped) = 0;
    new = logical(new);
    return
end

%%
function mapping = mapping_warpper(one, zero)
    function w = mapping_func(base)
        global is_real_data
        w = svdd(base);
        if is_real_data
            plotc(w)
        end
        return
    end

    mapping.one = mapping_func(one);
    mapping.zero = mapping_func(zero);
    return
end

%%
function out = mapping_result_of(bymapped, mapping_to)
    out = bymapped*mapping_to;
    out = out.data;
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

%%
function [] = plot_fig(train_x, train_y, test_x, test_y)

    ind1 = train_y==1;
    ind2 = train_y==0;
    
    figure
    hold on
    scatter(train_x(ind1,1),train_x(ind1,2), '+', 'b')
    scatter(train_x(ind2,1),train_x(ind2,2), '.', 'r')
    
    ind1 = test_y==1;
    ind2 = test_y==0;
    scatter(test_x(ind1,1),test_x(ind1,2), '+', 'black')
    scatter(test_x(ind2,1),test_x(ind2,2), '.', 'green')
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

