import torch
import torch.optim as optim
import pandas as pd
from src.Fleet_Assignment_Problem.models.pointer_net import MAPPOPolicy
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
import numpy as np

class MAPPOTrainer:
    def __init__(self, flight_dim=6, agent_dim=5, embed_dim=128, lr=3e-4, gamma=0.99, clip_ratio=0.2):
        self.policy = MAPPOPolicy(flight_dim, agent_dim, embed_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        
    def compute_gae(self, rewards, values, next_value, dones, lam=0.95):
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones
                next_val = next_value
            else:
                next_non_terminal = 1.0 - 0.0 
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * lam * next_non_terminal * last_gae_lam
        # Forcer l'alignement matériel des tenseurs sur le GPU
        advantages = advantages.to(self.device)
        if isinstance(values, list):
            values = torch.stack(values).to(self.device)
        else:
            values = values.to(self.device)
            
        returns = advantages + values
        return advantages, returns

    def train_step(self, rollouts, batch_size=32):
        obs_a_full = torch.stack([r['obs_a'] for r in rollouts])
        obs_f_full = torch.stack([r['obs_f'] for r in rollouts])
        pad_masks_full = torch.stack([r['pad_mask'] for r in rollouts])
        action_masks_full = torch.stack([r['masks'] for r in rollouts])
        actions_full = torch.stack([r['actions'] for r in rollouts])
        old_log_probs_full = torch.stack([r['log_probs'] for r in rollouts])
        returns_full = torch.stack([r['returns'] for r in rollouts])
        advantages_full = torch.stack([r['advantages'] for r in rollouts])
        
        advantages_full = (advantages_full - advantages_full.mean()) / (advantages_full.std() + 1e-8)
        
        dataset_size = len(rollouts)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices) 
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            
            obs_a = obs_a_full[batch_idx].to(self.device)
            obs_f = obs_f_full[batch_idx].to(self.device)
            pad_masks = pad_masks_full[batch_idx].to(self.device)
            action_masks = action_masks_full[batch_idx].to(self.device)
            actions = actions_full[batch_idx].to(self.device)
            old_log_probs = old_log_probs_full[batch_idx].to(self.device)
            returns = returns_full[batch_idx].to(self.device)
            advantages = advantages_full[batch_idx].to(self.device)
            
            log_probs, entropy, values = self.policy.evaluate_actions(
                obs_a, obs_f, actions, action_masks, pad_masks
            )
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages.unsqueeze(1)
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze(-1)).pow(2).mean()
            entropy_loss = entropy.mean()
            
            loss = actor_loss + critic_loss - 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

def collect_trajectories(env, trainer, physical_fleet_data, schedule_df, steps=128):
    rollouts = []
    
    airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}
    
    for ac in physical_fleet_data:
        ac['position'] = float(airport_to_idx.get("LFPO", 0.0))
    
    flights_data = []
    for _, row in schedule_df.iterrows():
        flights_data.append({
            'origin': float(airport_to_idx[row['From']]),
            'dest': float(airport_to_idx[row['To']]),
            'dep_time': row['Dept Time'].timestamp() / 60.0,
            'arr_time': row['Arr Time'].timestamp() / 60.0,
            'pax': float(row.get('flight_demand', 150)),
            'fare': float(row['Tarif'])
        })
        
    (obs_a, obs_f, pad_mask), masks = env.reset(physical_fleet_data, flights_data)
    
    for _ in range(steps):
        with torch.no_grad():
            actions, log_probs, _, value, _ = trainer.policy.get_action_and_value(
                obs_a.unsqueeze(0).to(trainer.device), 
                obs_f.unsqueeze(0).to(trainer.device), 
                masks.unsqueeze(0).to(trainer.device), 
                pad_mask.unsqueeze(0).to(trainer.device)
            )
            
        actions = actions.squeeze(0)
        log_probs = log_probs.squeeze(0)
        value = value.squeeze(0).squeeze(-1)
        
        next_obs_a, next_obs_f, next_pad_mask, next_masks, rewards, done = env.step(actions)
        
        rollouts.append({
            'obs_a': obs_a, 'obs_f': obs_f, 'pad_mask': pad_mask, 'masks': masks,
            'actions': actions, 'log_probs': log_probs, 'value': value,
            'rewards': rewards.mean(), 
        })
        
        obs_a, obs_f, pad_mask, masks = next_obs_a, next_obs_f, next_pad_mask, next_masks
        if done:
            break
            
    with torch.no_grad():
        _, _, _, next_value, _ = trainer.policy.get_action_and_value(
            obs_a.unsqueeze(0).to(trainer.device), 
            obs_f.unsqueeze(0).to(trainer.device), 
            masks.unsqueeze(0).to(trainer.device), 
            pad_mask.unsqueeze(0).to(trainer.device)
        )
        next_value = next_value.squeeze(0).squeeze(-1)
        
    rewards = torch.stack([r['rewards'] for r in rollouts])
    values = torch.stack([r['value'] for r in rollouts])
    
    advantages, returns = trainer.compute_gae(rewards, values, next_value, done)
    
    for i in range(len(rollouts)):
        rollouts[i]['advantages'] = advantages[i]
        rollouts[i]['returns'] = returns[i]
        
    return rollouts