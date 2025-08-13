classdef CentralizedDDGPAgent < handle
    properties
        actor, critic, actor_target, critic_target
        buffer
        obsDim, actDim, actLow, actHigh, actRange
        gamma = 0.99, tau = 0.005, batch_size = 128
    end
    
    methods
        function self = CentralizedDDGPAgent(obs_dim, act_dim, act_lows, act_highs)
            self.obsDim = obs_dim; self.actDim = act_dim;
            self.actor = FeedForwardNetwork([obs_dim, 256, 256, act_dim], {'relu','relu','tanh'});
            self.critic = FeedForwardNetwork([obs_dim + act_dim, 256, 256, 1], {'relu','relu','linear'});
            self.actor_target = FeedForwardNetwork([obs_dim, 256, 256, act_dim], {'relu','relu','tanh'});
            self.critic_target = FeedForwardNetwork([obs_dim + act_dim, 256, 256, 1], {'relu','relu','linear'});
            self.soft_update(1.0);
            self.buffer = ReplayBuffer(1e5);
            self.actLow = act_lows; self.actHigh = act_highs;
            self.actRange = act_highs - act_lows;
        end
        
        function action = getAction(self, obs, add_noise)
            [norm_action, ~] = self.actor.forward(obs);
            action = (norm_action + 1) / 2 .* self.actRange + self.actLow;
            if add_noise, action = action + 0.1 * self.actRange .* randn(size(action)); end
            action = max(self.actLow, min(self.actHigh, action));
        end
        
        function store(self, s, a, r, s2, d)
            self.buffer.add({s, a, r, s2, d});
        end
        
        function train(self)
            if ~self.buffer.is_ready(self.batch_size), return; end
            batch = self.buffer.sample(self.batch_size);
            s = cell2mat(cellfun(@(x) x{1}, batch, 'UniformOutput', false)');
            a = cell2mat(cellfun(@(x) x{2}, batch, 'UniformOutput', false)');
            r = cell2mat(cellfun(@(x) x{3}, batch, 'UniformOutput', false)');
            s2 = cell2mat(cellfun(@(x) x{4}, batch, 'UniformOutput', false)');
            d = cell2mat(cellfun(@(x) x{5}, batch, 'UniformOutput', false)');
            
            % Critic update
            [a2_norm, ~] = self.actor_target.forward(s2);
            a2 = (a2_norm + 1)/2 .* self.actRange + self.actLow;
            [q2, ~] = self.critic_target.forward([s2; a2]);
            y = r' + self.gamma * (1-d') .* q2;
            [q1, cache_c] = self.critic.forward([s; a]);
            [grads_c, ~] = self.critic.backward(q1 - y, cache_c);
            self.critic.update(grads_c);
            
            % Actor update
            [a1_norm, cache_a] = self.actor.forward(s);
            a1 = (a1_norm + 1)/2 .* self.actRange + self.actLow;
            [~, cache_c_for_a] = self.critic.forward([s; a1]);
            
            [~, d_input] = self.critic.backward(-ones(1, size(a1, 2)), cache_c_for_a);
            dQ_da = d_input(self.obsDim+1 : end, :);
            
            scaled_grad = dQ_da .* (self.actRange / 2);
            [grads_a, ~] = self.actor.backward(scaled_grad, cache_a);
            self.actor.update(grads_a);
            
            self.soft_update(self.tau);
        end
        
        function soft_update(self, tau)
            for i = 1:length(self.actor.W)
                self.actor_target.W{i} = tau*self.actor.W{i} + (1-tau)*self.actor_target.W{i};
                self.actor_target.b{i} = tau*self.actor.b{i} + (1-tau)*self.actor_target.b{i};
            end
            for i = 1:length(self.critic.W)
                self.critic_target.W{i} = tau*self.critic.W{i} + (1-tau)*self.critic_target.W{i};
                self.critic_target.b{i} = tau*self.critic.b{i} + (1-tau)*self.critic_target.b{i};
            end
        end
    end
end