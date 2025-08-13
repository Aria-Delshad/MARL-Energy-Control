classdef FeedForwardNetwork < handle
    properties
        layer_dims, activations
        W, b, mW, vW, mb, vb
        lr = 1e-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, t = 0
    end
    
    methods
        function self = FeedForwardNetwork(layer_dims, activations)
            self.layer_dims = layer_dims;
            self.activations = activations;
            self.initialize_parameters();
        end
        
        function initialize_parameters(self)
            num_layers = length(self.layer_dims);
            self.W = cell(1, num_layers-1); self.b = cell(1, num_layers-1);
            self.mW = cell(1, num_layers-1); self.vW = cell(1, num_layers-1);
            self.mb = cell(1, num_layers-1); self.vb = cell(1, num_layers-1);
            for i = 1:num_layers-1
                in_dim = self.layer_dims(i); out_dim = self.layer_dims(i+1);
                limit = sqrt(6 / (in_dim + out_dim));
                self.W{i} = (rand(out_dim, in_dim) * 2 - 1) * limit;
                self.b{i} = zeros(out_dim, 1);
                self.mW{i} = zeros(size(self.W{i})); self.vW{i} = zeros(size(self.W{i}));
                self.mb{i} = zeros(size(self.b{i})); self.vb{i} = zeros(size(self.b{i}));
            end
        end
        
        function [A, cache] = forward(self, X)
            cache = struct('A', {cell(1, length(self.W) + 1)}, 'Z', {cell(1, length(self.W))});
            cache.A{1} = X;
            A = X;
            for i = 1:length(self.W)
                Z = self.W{i} * A + self.b{i};
                activation_func = self.activations{i};
                if strcmp(activation_func, 'relu'), A = max(0, Z);
                elseif strcmp(activation_func, 'tanh'), A = tanh(Z);
                elseif strcmp(activation_func, 'linear'), A = Z; end
                cache.Z{i} = Z; cache.A{i+1} = A;
            end
        end
        
        function [grads, dA_input] = backward(self, dAL, cache)
            grads = struct('dW', {cell(1, length(self.W))}, 'db', {cell(1, length(self.b))});
            dA = dAL;
            for i = length(self.W):-1:1
                A_prev = cache.A{i}; Z = cache.Z{i};
                activation_func = self.activations{i};
                if strcmp(activation_func, 'relu'), dZ = dA .* (Z > 0);
                elseif strcmp(activation_func, 'tanh'), dZ = dA .* (1 - tanh(Z).^2);
                elseif strcmp(activation_func, 'linear'), dZ = dA; end
                m = size(A_prev, 2);
                grads.dW{i} = (1/m) * (dZ * A_prev');
                grads.db{i} = (1/m) * sum(dZ, 2);
                dA = self.W{i}' .* dZ; 
            end
            dA_input = dA; 
        end
        
        function update(self, grads)
            self.t = self.t + 1;
            for i = 1:length(self.W)
                self.mW{i} = self.beta1 * self.mW{i} + (1-self.beta1) * grads.dW{i};
                self.vW{i} = self.beta2 * self.vW{i} + (1-self.beta2) * (grads.dW{i}.^2);
                mW_hat = self.mW{i} / (1 - self.beta1^self.t);
                vW_hat = self.vW{i} / (1 - self.beta2^self.t);
                self.W{i} = self.W{i} - self.lr * mW_hat ./ (sqrt(vW_hat) + self.epsilon);
                
                self.mb{i} = self.beta1 * self.mb{i} + (1-self.beta1) * grads.db{i};
                self.vb{i} = self.beta2 * self.vb{i} + (1-self.beta2) * (grads.db{i}.^2);
                mb_hat = self.mb{i} / (1 - self.beta1^self.t);
                vb_hat = self.vb{i} / (1 - self.beta2^self.t);
                self.b{i} = self.b{i} - self.lr * mb_hat ./ (sqrt(vb_hat) + self.epsilon);
            end
        end
    end
end