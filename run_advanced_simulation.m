clear; clc; close all;
rng(42);

% --- Experiment Settings ---
episodes = 50; 
saveDir = "outputs_advanced";
if ~exist(saveDir, 'dir'); mkdir(saveDir); end
fprintf('--- Advanced MARL Simulation ---\n');

% --- Run Baselines ---
fprintf('\n1. Running Heuristic Baseline...\n');
[ATD_h, TEC_h, traj_h] = run_heuristic_baseline();
plot_trajectories(traj_h, 'Heuristic Baseline', fullfile(saveDir, 'traj_heuristic.png'));

fprintf('2. Running User-Only Baseline...\n');
[ATD_u, TEC_u, traj_u] = run_user_only_baseline();
plot_trajectories(traj_u, 'User-Only Baseline', fullfile(saveDir, 'traj_user_only.png'));

% --- Run Centralized DDPG ---
fprintf('\n3. Training Centralized DDPG Agent...\n');
env_c = CommunityEnv();
agent_c = CentralizedDDGPAgent(env_c.obsDim, env_c.actDim, env_c.actLows, env_c.actHighs);
rewards_c = zeros(episodes, 1);
for ep = 1:episodes
    obs = env_c.reset();
    ep_reward = 0;
    for k = 1:env_c.K
        action = agent_c.getAction(obs, true); 
        [next_obs, reward_vec, done] = env_c.step_global_action(action);
        agent_c.store(obs, action, sum(reward_vec), next_obs, done);
        agent_c.train();
        obs = next_obs;
        ep_reward = ep_reward + sum(reward_vec);
    end
    rewards_c(ep) = ep_reward;
    fprintf('Centralized DDPG | Episode: %d/%d | Reward: %.2f\n', ep, episodes, ep_reward);
end
[ATD_c, TEC_c, traj_c] = evaluate_agent(env_c, agent_c);
plot_trajectories(traj_c, 'Centralized DDPG', fullfile(saveDir, 'traj_centralized.png'));

% --- Run Proposed MADDPG ---
fprintf('\n4. Training Proposed MADDPG Agent...\n');
env_m = CommunityEnv();
agent_m = MADDPGAgent(env_m);
rewards_m = zeros(episodes, 1);
for ep = 1:episodes
    [local_obs, global_obs] = env_m.reset_maddpg();
    ep_reward = 0;
    for k = 1:env_m.K
        actions = agent_m.getActions(local_obs, true); % Decentralized actions with noise
        [next_local, next_global, reward_vec, done] = env_m.step_maddpg_actions(actions);
        agent_m.store(global_obs, cell2mat(actions'), reward_vec, next_global, done);
        agent_m.train();
        local_obs = next_local;
        global_obs = next_global;
        ep_reward = ep_reward + sum(reward_vec);
    end
    rewards_m(ep) = ep_reward;
    fprintf('Proposed MADDPG | Episode: %d/%d | Reward: %.2f\n', ep, episodes, ep_reward);
end
[ATD_m, TEC_m, traj_m] = evaluate_agent_maddpg(env_m, agent_m);
plot_trajectories(traj_m, 'Proposed MADDPG', fullfile(saveDir, 'traj_maddpg.png'));

% --- Final Results & Plots ---
fprintf('\n--- Simulation Complete. Final Results: ---\n');
results = table;
results.Method = {'Heuristic'; 'User-Only'; 'Centralized DDPG'; 'Proposed MADDPG'};
results.ATD = [ATD_h; ATD_u; ATD_c; ATD_m];
results.TEC = [TEC_h; TEC_u; TEC_c; TEC_m];
disp(results);
writetable(results, fullfile(saveDir, 'results_summary.csv'));

% Plot training rewards
figure('Name','Training Comparison','Visible','off');
plot(rewards_c, 'LineWidth', 1.5, 'DisplayName', 'Centralized DDPG');
hold on;
plot(rewards_m, 'LineWidth', 1.5, 'DisplayName', 'Proposed MADDPG');
title('Training Rewards'); xlabel('Episode'); ylabel('Total Reward');
legend; grid on;
saveas(gcf, fullfile(saveDir, 'training_rewards.png'));
fprintf('Results and plots saved to "%s" folder.\n', saveDir);

%% Helper Functions for baselines and evaluation
function [ATD, TEC, traj] = run_heuristic_baseline()
    env = CommunityEnv();
    obs = env.reset();
    for k = 1:env.K
        % Heuristic logic
        pk = env.price(env.k);
        c_k = env.p.c_max * (pk < env.pbar);
        P_d = ones(env.N, 1); P_g = ones(env.N, 1);
        if env.Tout(env.k) > env.p.T_target, P_d = -P_d; P_g = -P_g; end
        actions = [reshape([P_g, P_d]', [], 1); c_k];
        env.step_global_action(actions);
    end
    [ATD, TEC] = env.compute_metrics();
    traj = env.history;
end

function [ATD, TEC, traj] = run_user_only_baseline()
    env = CommunityEnv();
    obs = env.reset();
    Kp = 0.5;
    for k = 1:env.K
        err = env.p.T_target - env.Tin;
        P_g = min(env.p.Pg_max, max(-env.p.Pg_max, Kp * err .* env.p.C));
        P_d = zeros(env.N, 1); c_k = 0;
        actions = [reshape([P_g, P_d]', [], 1); c_k];
        env.step_global_action(actions);
    end
    [ATD, TEC] = env.compute_metrics();
    traj = env.history;
end

function [ATD, TEC, traj] = evaluate_agent(env, agent)
    obs = env.reset();
    for k = 1:env.K
        action = agent.getAction(obs, false); % No noise for evaluation
        obs = env.step_global_action(action);
    end
    [ATD, TEC] = env.compute_metrics();
    traj = env.history;
end

function [ATD, TEC, traj] = evaluate_agent_maddpg(env, agent)
    [local_obs, ~] = env.reset_maddpg();
    for k = 1:env.K
        actions = agent.getActions(local_obs, false);
        [local_obs, ~] = env.step_maddpg_actions(actions);
    end
    [ATD, TEC] = env.compute_metrics();
    traj = env.history;
end

function plot_trajectories(traj, title_str, save_path)
    K = size(traj.SOC, 2);
    t = 1:K;
    f = figure('Name', title_str, 'Visible','off', 'Position', [100, 100, 900, 700]);
    
    subplot(3,1,1);
    plot(t, traj.price, 'k--', 'DisplayName', 'Price'); hold on;
    plot(t, traj.Tout, 'r-', 'DisplayName', 'T_{out}');
    title([title_str, ' - Externals']); ylabel('Price / Temp'); grid on; legend;
    
    subplot(3,1,2);
    plot(t, traj.Tin', 'LineWidth', 1.5); hold on;
    yline(traj.T_target, 'k--', 'DisplayName', 'Target');
    title('Indoor Temperatures'); ylabel('Temp (°C)'); grid on;
    
    subplot(3,1,3);
    plot(t, traj.SOC, 'g-', 'LineWidth', 2, 'DisplayName', 'SESS SOC');
    title('SESS State of Charge'); ylabel('SOC (kWh)'); xlabel('Time (h)'); grid on;
    
    saveas(f, save_path);
    close(f);
end