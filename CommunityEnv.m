classdef CommunityEnv < handle
    properties
        N = 2; K = 24;
        p; % parameters
        
        % State variables
        k; Tin; SOC; pbar;
        price; Tout;
        
        % Dimensions for RL
        obsDim; actDim;
        actLows; actHighs;
        
        % History for plotting
        history;
    end
    
    methods
        function self = CommunityEnv()
           
            self.p = self.get_default_params();
            

            self.obsDim = 4 * self.N + 4;
            
            
            self.actDim = 2 * self.N + 1;
            
            
            Pg_max = self.p.Pg_max; Pd_max = self.p.Pd_max; c_max = self.p.c_max;
            self.actLows = [repmat([-Pg_max; -Pd_max], self.N, 1); 0];
            self.actHighs = [repmat([Pg_max; Pd_max], self.N, 1); c_max];
            
            [self.price, self.Tout] = self.make_profiles('spring', self.K);
            self.reset();
        end
        
        function obs = reset(self)
            self.k = 1;
            self.Tin = self.p.T_target + 1.0 * ones(self.N, 1);
            self.SOC = 0.5 * self.p.SOC_max;
            self.pbar = self.price(1);
            self.init_history();
            obs = self.get_global_observation();
        end
        
        function [local_obs, global_obs] = reset_maddpg(self)
            global_obs = self.reset();
            local_obs = self.get_local_observations();
        end
        
        function [next_obs, reward_vec, done] = step_global_action(self, action)
            % For centralized agent
            a_b = reshape(action(1:2*self.N), 2, self.N)';
            a_e = action(end);
            [next_obs, reward_vec, done] = self.step(a_b, a_e);
        end
        
        function [next_local, next_global, reward_vec, done] = step_maddpg_actions(self, actions)
            % For MADDPG agent
            a_b = cell2mat(actions(1:self.N));
            a_e = actions{self.N+1};
            [next_global, reward_vec, done] = self.step(a_b, a_e);
            next_local = self.get_local_observations();
        end
        
        function [ATD, TEC] = compute_metrics(self)
            Tin_hist = self.history.Tin;
            Pg_hist = self.history.Pg;
            duse_hist = self.history.duse;
            
            ATD = mean(abs(Tin_hist - self.p.T_target), 'all');
            TEC = sum(abs(Pg_hist), 'all') + sum(duse_hist, 'all');
        end
    end
    
    methods (Access = private)
        function [next_obs, reward_vec, done] = step(self, a_buildings, a_sess)
            
            self.log_history(a_buildings(:,1), a_buildings(:,2), a_sess, zeros(self.N,1)); % pre-calculate d_i later
            
            pk = self.price(self.k);
            pbar_prev = self.pbar;
            
            
            P_g = min(self.p.Pg_max, max(-self.p.Pg_max, a_buildings(:,1)));
            P_d_req = min(self.p.Pd_max, max(-self.p.Pd_max, a_buildings(:,2)));
            c_k = min(self.p.c_max, max(0, a_sess));
            
            
            d_i = min(self.p.d_max, abs(P_d_req) / self.p.delta_d);
            avail = self.SOC + self.p.delta_c * c_k;
            if sum(d_i) > avail && sum(d_i) > 1e-6
                d_i = d_i * (avail / sum(d_i));
            end
            P_d = sign(P_d_req) .* d_i * self.p.delta_d;

            % Update state
            SOC_next = self.SOC + self.p.delta_c * c_k - sum(d_i);
            Tin_next = zeros(self.N, 1);
            for i = 1:self.N
                Tin_next(i) = self.p.d1(i)*self.Tin(i) + self.p.d2(i)*P_d(i) ...
                    + self.p.d3(i)*P_g(i) + self.p.d4(i)*self.Tout(self.k);
            end
            pbar_next = (1-self.p.eta)*self.pbar + self.p.eta*pk;
            
            % Rewards
            r_b = -(self.p.alpha_temp * abs(Tin_next - self.p.T_target) + self.p.alpha_energy * pk .* abs(P_g));
            r_e = (pbar_prev - pk) * c_k;
            
            % Commit state for next step
            self.Tin = Tin_next;
            self.SOC = SOC_next;
            self.pbar = pbar_next;
            
            
            self.k = self.k + 1;
            done = (self.k > self.K);

            if done
                r_e = r_e - self.p.beta * self.SOC;
            end
            reward_vec = [r_b; r_e];
            
            
            self.history.Pd(:, self.k-1) = P_d;
            self.history.duse(:, self.k-1) = d_i;
            
            next_obs = self.get_global_observation();
        end

        function obs = get_global_observation(self)
            local_obs_list = self.get_local_observations();
            all_local_vectors = vertcat(local_obs_list{:});
            obs = [all_local_vectors; self.pbar];
        end

        function local_obs = get_local_observations(self)
            local_obs = cell(self.N + 1, 1);
            safe_k = min(self.k, self.K); 
            
            for i = 1:self.N
                local_obs{i} = [self.Tin(i); self.Tout(safe_k); self.price(safe_k); self.SOC];
            end
            local_obs{self.N+1} = [self.Tout(safe_k); self.price(safe_k); self.SOC];
        end
        
        function init_history(self)
            self.history.Tin = zeros(self.N, self.K);
            self.history.SOC = zeros(1, self.K);
            self.history.Pg = zeros(self.N, self.K);
            self.history.Pd = zeros(self.N, self.K);
            self.history.duse = zeros(self.N, self.K);
            self.history.c = zeros(1, self.K);
            self.history.price = self.price';
            self.history.Tout = self.Tout';
            self.history.T_target = self.p.T_target;
        end
        
        function log_history(self, P_g, P_d, c_k, d_i)
            % Log the state at the beginning of timestep k
            self.history.Tin(:, self.k) = self.Tin;
            self.history.SOC(self.k) = self.SOC;
            self.history.Pg(:, self.k) = P_g;
            self.history.Pd(:, self.k) = P_d;
            self.history.c(self.k) = c_k;
            self.history.duse(:, self.k) = d_i;
        end
        
        function p = get_default_params(~)
            p.T_target=22; p.R=[8; 10]; p.C=[15; 18];
            p.wd=[0.9; 1.0]; p.wg=[1.1; 1.0];
            p.Pg_max=5; p.Pd_max=5; p.SOC_max=50; p.c_max=10; p.d_max=5;
            p.delta_c=0.9; p.delta_d=1.1; p.alpha_temp=10; p.alpha_energy=1;
            p.gamma=0.99; p.eta=0.2; p.beta=0.5;
            p.d1=0.9*ones(2,1); p.d4=0.1*ones(2,1);
            p.d2=p.wd./p.C; p.d3=p.wg./p.C;
        end
        
        function [price, Tout] = make_profiles(~, season, K)
            t = (1:K)';
            switch season
                case 'winter'
                    Tout = 5 + 4*sin(2*pi*(t-6)/24); price = 0.2 + 0.1*sin(2*pi*(t-10)/24);
                case 'summer'
                    Tout = 28 + 6*sin(2*pi*(t-8)/24); price = 0.15 + 0.1*sin(2*pi*(t-10)/24);
                otherwise % spring
                    Tout = 18 + 5*sin(2*pi*(t-8)/24); price = 0.1 + 0.05*sin(2*pi*(t-10)/24);
            end
        end
    end
end