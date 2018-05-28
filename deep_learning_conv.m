function y = deep_learning_conv(x, w)

[w_row, w_col, n_filter] = size(w);

[x_row, x_col, ~] = size(x);

y_row = x_row - w_row + 1;
y_col = x_col - w_col + 1;

y = zeros(y_row, y_col, n_filter);

for k = 1:n_filter
    filter = w(:, :, k);
    filter = rot90(filter, 2);
    y(:, :, k) = conv2(x, filter, 'valid');
end

end