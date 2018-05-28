function [y, y_5, y_4, y_3, y_2] = cnn_forward(x, w_1, w_5, w_o)

y_1 = deep_learning_conv(x, w_1);
y_2 = relu(y_1);

y_3 = deep_learning_pool(y_2);
y_4 = y_3(:);

v_5 = w_5 * y_4;
y_5 = relu(v_5);

v = w_o * y_5;
y = soft_max(v);

end
