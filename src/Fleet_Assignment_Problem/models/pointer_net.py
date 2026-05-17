import torch
import torch.nn as nn
import torch.nn.functional as F

class FlightEncoder(nn.Module):
    def __init__(self, flight_dim, embed_dim, n_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(flight_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, flights, pad_mask=None):
        x = self.input_proj(flights)
        return self.transformer(x, src_key_padding_mask=pad_mask)

class PointerActor(nn.Module):
    def __init__(self, agent_dim, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(agent_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, agent_state, flight_embeddings, action_mask):
        Q = self.q_proj(agent_state).unsqueeze(1) 
        K = self.k_proj(flight_embeddings)
        
        logits = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) * self.scale
        
        # Verrouillage physique différentiable
        logits = logits.masked_fill(~action_mask.bool(), float('-inf'))
        return logits

class CentralizedCritic(nn.Module):
    def __init__(self, agent_dim, flight_embed_dim, hidden_dim):
        super().__init__()
        self.agent_proj = nn.Linear(agent_dim, hidden_dim)
        self.pool_agents = nn.AdaptiveAvgPool1d(1)
        self.pool_flights = nn.AdaptiveAvgPool1d(1)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + flight_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, all_agents_state, flight_embeddings):
        a_emb = F.relu(self.agent_proj(all_agents_state))
        
        a_pooled = self.pool_agents(a_emb.transpose(1, 2)).squeeze(-1)
        f_pooled = self.pool_flights(flight_embeddings.transpose(1, 2)).squeeze(-1)
        
        global_state = torch.cat([a_pooled, f_pooled], dim=-1)
        return self.value_head(global_state)

class MAPPOPolicy(nn.Module):
    def __init__(self, flight_dim, agent_dim, embed_dim=128):
        super().__init__()
        self.flight_encoder = FlightEncoder(flight_dim, embed_dim)
        self.actor = PointerActor(agent_dim, embed_dim)
        self.critic = CentralizedCritic(agent_dim, embed_dim, embed_dim)

    def get_action_and_value(self, agents_state, flights_state, action_masks, flights_pad_mask=None):
        flight_emb = self.flight_encoder(flights_state, pad_mask=flights_pad_mask)
        
        logits = []
        for i in range(agents_state.size(1)):
            logits.append(self.actor(agents_state[:, i], flight_emb, action_masks[:, i]))
        logits = torch.stack(logits, dim=1) 
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        value = self.critic(agents_state, flight_emb)
        
        return actions, log_probs, dist.entropy(), value, flight_emb

    def evaluate_actions(self, agents_state, flights_state, actions, action_masks, flights_pad_mask=None):
        flight_emb = self.flight_encoder(flights_state, pad_mask=flights_pad_mask)
        
        logits = []
        for i in range(agents_state.size(1)):
            logits.append(self.actor(agents_state[:, i], flight_emb, action_masks[:, i]))
        logits = torch.stack(logits, dim=1)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        value = self.critic(agents_state, flight_emb)
        
        return log_probs, entropy, value