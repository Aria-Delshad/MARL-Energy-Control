classdef MADDPGAgent < handle
    properties
        agents, n_agents, buffer
        gamma = 0.99, tau = 0.005, batch_size = 128
        local_obs_dims, act_dims
    end
    
    methods
        function self = MADDPGAgent(env)
            self.n_agents = env.N + 1;
            self.agents = cell(self.n_agents, 1);
            self.buffer = ReplayBuffer(1e5);
            
            self.act_dims = [repmat(2, env.N, 1); 1];
            self.local_obs_dims = [repmat(4, env.N, 1); 3];
            global_obs_dim = sum(self.local_obs_dims) + 1;
            global_act_dim = sum(self.act_dims);
            
            for i = 1:self.n_agents
                agent.id = i;
                if i <= env.N
                    low = [-env.p.Pg_max; -env.p.Pd_max]; high = [env.p.Pg_max; env.p.Pd_max];
                else
                    low = 0; high = env.p.c_max;
                end
                
                agent.actor = FeedForwardNetwork([self.local_obs_dims(i), 128, 128, self.act_dims(i)], {'relu','relu','tanh'});
                agent.critic = FeedForwardNetwork([global_obs_dim + global_act_dim, 256, 256, 1], {'relu','relu','linear'});
                agent.actor_target = FeedForwardNetwork([self.local_obs_dims(i), 128, 128, self.act_dims(i)], {'relu','relu','tanh'});
                agent.critic_target = FeedForwardNetwork([global_obs_dim + global_act_dim, 256, 256, 1], {'relu','relu','linear'});
                
                agent.actor_target.W = agent.actor.W; agent.actor_target.b = agent.actor.b;
                agent.critic_target.W = agent.critic.W; agent.critic_target.b = agent.critic.b;
                
                agent.actLow = low; agent.actHigh = high;
                agent.actRange = high-low;
                
                self.agents{i} = agent;
            end
        end
        
        function actions = getActions(self, local_obs, add_noise)
            actions = cell(self.n_agents, 1);
            for i = 1:self.n_agents
                agent = self.agents{i};
                [norm_a, ~] = agent.actor.forward(local_obs{i});
                action = (norm_a + 1)/2 .* agent.actRange + agent.actLow;
                if add_noise, action = action + 0.1 * agent.actRange .* randn(size(action)); end
                actions{i} = max(agent.actLow, min(agent.actHigh, action));
            end
        end
        
        function store(self, s, a, r, s2, d)
            self.buffer.add({s, a, r, s2, d});
        end
        
        function train(self)
            if ~self.buffer.is_ready(self.batch_size), return; end
            
            batch = self.buffer.sample(self.batch_size);
            s_batch = cell2mat(cellfun(@(x) x{1}, batch, 'UniformOutput', false)');
            a_batch = cell2mat(cellfun(@(x) x{2}, batch, 'UniformOutput', false)');
            r_batch = cell2mat(cellfun(@(x) x{3}, batch, 'UniformOutput', false)');
            s2_batch = cell2mat(cellfun(@(x) x{4}, batch, 'UniformOutput', false)');
            d_batch = cell2mat(cellfun(@(x) x{5}, batch, 'UniformOutput', false)');
            
            s2_split = mat2cell(s2_batch(1:sum(self.local_obs_dims),:), self.local_obs_dims, self.batch_size);
            a2_all = [];
            for i = 1:self.n_agents
                agent_i = self.agents{i};
                [a2_norm_i, ~] = agent_i.actor_target.forward(s2_split{i});
                a2_i = (a2_norm_i + 1)/2 .* agent_i.actRange + agent_i.actLow;
                a2_all = [a2_all; a2_i];
            end
            
            s_split = mat2cell(s_batch(1:sum(self.local_obs_dims),:), self.local_obs_dims, self.batch_size);
            a1_all_online = [];
            caches_a = cell(self.n_agents, 1);
            for j = 1:self.n_agents
                agent_j = self.agents{j};
                [a1_norm_j, cache_a_j] = agent_j.actor.forward(s_split{j});
                caches_a{j} = cache_a_j;
                a1_j = (a1_norm_j + 1)/2 .* agent_j.actRange + agent_j.actLow;
                a1_all_online = [a1_all_online; a1_j];
            end

            for i = 1:self.n_agents
                agent = self.agents{i};
                
                % Critic update
                [q2, ~] = agent.critic_target.forward([s2_batch; a2_all]);
                y = r_batch(i,:)' + self.gamma * (1-d_batch') .* q2;
                [q1, cache_c] = agent.critic.forward([s_batch; a_batch]);
                [grads_c, ~] = agent.critic.backward(q1 - y, cache_c);
                agent.critic.update(grads_c);
                
                % Actor update for agent i
                current_actions_for_grad = a_batch; % Use actions from buffer
                [a1_norm_i, ~] = agent.actor.forward(s_split{i});
                a1_i = (a1_norm_i + 1)/2 .* agent.actRange + agent.actLow;
                
                act_start_idx = sum(self.act_dims(1:i-1)) + 1;
                act_end_idx = act_start_idx + self.act_dims(i) - 1;
                current_actions_for_grad(act_start_idx:act_end_idx, :) = a1_i;
                
                [~, cache_c_for_a] = agent.critic.forward([s_batch; current_actions_for_grad]);
                [~, d_input] = agent.critic.backward(-ones(1, size(current_actions_for_grad, 2)), cache_c_for_a);
                
                global_obs_dim = size(s_batch, 1);
                dQ_da_i = d_input(global_obs_dim + act_start_idx : global_obs_dim + act_end_idx, :);

                scaled_grad = dQ_da_i .* (agent.actRange / 2);
                [grads_a, ~] = agent.actor.backward(scaled_grad, caches_a{i});
                agent.actor.update(grads_a);
            end
            
            for i = 1:self.n_agents
                self.soft_update_agent(i);
            end
        end
        
        function soft_update_agent(self, agent_idx)
            agent = self.agents{agent_idx};
            for i = 1:length(agent.actor.W)
                agent.actor_target.W{i} = self.tau*agent.actor.W{i} + (1-self.tau)*agent.actor_target.W{i};
                agent.actor_target.b{i} = self.tau*agent.actor.b{i} + (1-self.tau)*agent.actor_target.b{i};
            end
            for i = 1:length(agent.critic.W)
                 agent.critic_target.W{i} = self.tau*agent.critic.W{i} + (1-self.tau)*agent.critic_target.W{i};
                 agent.critic_target.b{i} = self.tau*agent.critic.b{i} + (1-self.tau)*agent.critic_target.b{i};
            end
        end
    end
end