function test_mnist(varargin)

if nargin == 0 || (nargin > 0 && varargin{1} == 0)
    
    images = loadMNISTImages('t10k-images-idx3-ubyte');
    images = reshape(images, 28, 28, []);
    
    labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    labels(labels == 0) = 10; % 0 -> 10
    
    training_data = images(:, :, 1:8000);
    targets = labels(1:8000);
    
    rng default
    
    w_1 = 1e-2 * randn(9, 9, 20);
    w_5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
    w_o = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);
    
    for epoch = 1:10
        disp(epoch);
        [w_1, w_5, w_o] = cnn_train(w_1, w_5, w_o, training_data, targets);
    end
    
    save('mnist_conv.mat');
    
else
    load('mnist_conv.mat');
end

interference_data = images(:, :, 8001:10000);
real_answer = labels(8001:10000);

acc = 0;
N = length(real_answer);

for k = 1:N
    y = cnn_forward(interference_data(:, :, k), w_1, w_5, w_o);
    
    [~, ii] = max(y);
    
    if ii == real_answer(k)
        acc = acc + 1;
    end
end

acc = acc / N;
fprintf('accuracy is %f.\n', acc);

end