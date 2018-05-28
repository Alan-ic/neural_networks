function [w_1, w_5, w_o] = cnn_train(w_1, w_5, w_o, training_data, training_target)

alpha = 0.01;
beta = 0.95;

momentum_1 = zeros(size(w_1));
momentum_5 = zeros(size(w_5));
momentum_o = zeros(size(w_o));

N = length(training_target);

batch_size = 100;
batch_list = 1:batch_size:(N-batch_size+1);

% epoch loop
for batch = 1:length(batch_list)
    dw_1 = zeros(size(w_1));
    dw_5 = zeros(size(w_5));
    dw_o = zeros(size(w_o));
    
    % mini-batch loop
    begin = batch_list(batch);
    for k = begin:begin+batch_size-1
        % forward pass
        x = training_data(:, :, k);
        [y, y_5, y_4, y_3, y_2] = cnn_forward(x, w_1, w_5, w_o);
        
        % back propagation
        
        % one-hot encoding
        d = zeros(10, 1);
        d(sub2ind(size(d), training_target(k), 1)) = 1;
        
        error_value = d - y;
        delta = error_value;
        
        error_5 = w_o.' * delta;
        delta_5 = (y_5 > 0) .* error_5;
        
        error_4 = w_5.' * delta_5;
        
        error_3 = reshape(error_4, size(y_3));
        w_3 = ones(size(y_2)) / (2 * 2);
        
        error_2 = zeros(20, 20, 20);
        for index = 1:20
            error_2(:, :, index) = kron(error_3(:, :, index), ones(2, 2)) .* w_3(:, :, index);
        end
        
        delta_2 = (y_2 > 0) .* error_2; % ReLU layer
        
        delta_1 = zeros(size(w_1));
        for index = 1:20
            delta_1(:, :, index) = conv2(x(:, :), rot90(delta_2(:, :, index), 2), 'valid');
        end
        
        dw_1 = dw_1 + delta_1;
        dw_5 = dw_5 + delta_5 * y_4.';
        dw_o = dw_o + delta * y_5.';
        
        % update weights
        dw_1 = dw_1 / batch_size;
        dw_5 = dw_5 / batch_size;
        dw_o = dw_o / batch_size;
        
        momentum_1 = alpha * dw_1 + beta * momentum_1;
        w_1 = w_1 + momentum_1;
        
        momentum_5 = alpha * dw_5 + beta * momentum_5;
        w_5 = w_5 + momentum_5;
        
        momentum_o = alpha * dw_o + beta * momentum_o;
        w_o = w_o + momentum_o;
    end
end

end
        
        
        