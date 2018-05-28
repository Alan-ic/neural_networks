function p = soft_max(q)

q = q(:);
n = length(q);
p = zeros(n, 1);

dominant = sum(exp(q));

p = exp(q) ./ dominant;

end