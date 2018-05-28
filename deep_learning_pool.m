function y = deep_learning_pool(x)
% 2 * 2 mean pooling

[row, col, n_filter] = size(x);

y = zeros(row/2, col/2, n_filter);
for k = 1:n_filter
    filter = ones(2, 2) / (2 * 2);
    image = conv2(x(:, :, k), filter, 'valid');
    y(:, :, k) = image(1:2:end, 1:2:end);
end

end