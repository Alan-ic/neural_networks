function xor_gate_simulate
% a simple neural network simulates XOR gate
x_1 = [0, 0; ...
0, 1; ...
1, 0; ...
1, 1];

y_1 = [0; 1; 1; 0];

[trained_weight_1, trained_weight_2] = train(x_1, y_1);

for index = 1:size(x_1, 1)
    input = x_1(index, :);
    prediction = predict(input, trained_weight_1, trained_weight_2);
    disp([num2str(input), '  ', num2str(prediction)]);
end

end

function [weight_layer_1, weight_layer_2] = train(x, y)

% initialize the weights
rng('default')

N = 16;

weight_layer_1 = 2 * rand(size(x, 2), N) - 1;
weight_layer_2 = 2 * rand(N, 1) - 1;

% train the neural network
for training_time = 1:100000
    layer_1_output = 1 ./ (1 + exp(-(x * weight_layer_1)));
    layer_2_output = 1 ./ (1 + exp(-(layer_1_output * weight_layer_2)));

    % back propagation
    layer_2_delta = (y - layer_2_output) .* (layer_2_output .* (1 - layer_2_output));
    layer_1_delta = layer_2_delta * weight_layer_2.' .* (layer_1_output .* (1 - layer_1_output));

    % correct the weights
    weight_layer_2 = weight_layer_2 + layer_1_output.' * layer_2_delta;
    weight_layer_1 = weight_layer_1 + x.' * layer_1_delta;
end

end

function output = predict(x, weight_1, weight_2)
    layer_1_output = 1 ./ (1 + exp(-(x * weight_1)));
    output = 1 ./ (1 + exp(-(layer_1_output * weight_2)));
end
