import torch
import torch.optim as optim
from pointer_net import MAPPOPolicy
from fap_ma_env import FAPParallelEnv
import numpy as np
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule

class MAPPOTrainer:
    def __init__(self, flight_dim=6, agent_dim=5, embed_dim=128, lr=3e-4, gamma=0.99, clip_ratio=0.2):
        self.policy = MAPPOPolicy(flight_dim, agent_dim, embed_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
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
        returns = advantages + values
        return advantages, returns

    def train_step(self, rollouts):
        obs_a = torch.stack([r['obs_a'] for r in rollouts])
        obs_f = torch.stack([r['obs_f'] for r in rollouts])
        pad_masks = torch.stack([r['pad_mask'] for r in rollouts])
        action_masks = torch.stack([r['masks'] for r in rollouts])
        actions = torch.stack([r['actions'] for r in rollouts])
        old_log_probs = torch.stack([r['log_probs'] for r in rollouts])
        returns = torch.stack([r['returns'] for r in rollouts])
        advantages = torch.stack([r['advantages'] for r in rollouts])
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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

def collect_trajectories(env, trainer, physical_fleet_data, steps=128):
    rollouts = []
    
    # 1. Extraction du paramètre N à partir de la flotte déployée
    num_aircraft = len(physical_fleet_data)
    
    # 2. Génération du planning stochastique M couplé au nombre d'avions
    schedule_df = generate_dynamic_schedule(num_aircraft)
    
    # 3. Projection du DataFrame dans l'espace d'observation de l'environnement
    flights_data = []
    for _, row in schedule_df.iterrows():
        flights_data.append({
            'origin': row['From'],
            'dest': row['To'],
            'dep_time': row['Dept Time'].timestamp() / 60.0,
            'arr_time': row['Arr Time'].timestamp() / 60.0,
            'pax': 150, # Intégration future de l'inférence LSTM (inference_ppo.py)
            'fare': row['Tarif']
        })
        
    # 4. Écrasement des dimensions de l'épisode
    obs_a, obs_f, pad_mask = env.reset(physical_fleet_data, flights_data)
    masks = env._get_masks()
    
    for _ in range(steps):
        with torch.no_grad():
            actions, log_probs, _, value, _ = trainer.policy.get_action_and_value(
                obs_a.unsqueeze(0), obs_f.unsqueeze(0), masks.unsqueeze(0), pad_mask.unsqueeze(0)
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
            obs_a.unsqueeze(0), obs_f.unsqueeze(0), masks.unsqueeze(0), pad_mask.unsqueeze(0)
        )
        next_value = next_value.squeeze(0).squeeze(-1)
        
    rewards = torch.stack([r['rewards'] for r in rollouts])
    values = torch.stack([r['value'] for r in rollouts])
    
    advantages, returns = trainer.compute_gae(rewards, values, next_value, done)
    
    for i in range(len(rollouts)):
        rollouts[i]['advantages'] = advantages[i]
        rollouts[i]['returns'] = returns[i]
        
    return rollouts