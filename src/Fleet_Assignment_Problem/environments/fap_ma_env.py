import numpy as np
import torch

class FAPParallelEnv:
    def __init__(self, num_airports, max_flights=100, spill_penalty_coef=1.5):
        self.num_airports = num_airports
        self.max_flights = max_flights
        self.spill_penalty_coef = spill_penalty_coef
        self.agents = []
        self.flights = None 
        self.num_agents = 0
        self.num_flights = 0

    def reset(self, fleet_data, flights_data):
        self.num_agents = len(fleet_data)
        self.num_flights = len(flights_data)
        
        self.agents = np.array([
            [
                f['position'], 
                0.0, 
                f['capacity'], 
                f['speed'], 
                f['cost']
            ] for f in fleet_data
        ], dtype=np.float32)
        
        self.flights = np.array([
            [
                fl['origin'], 
                fl['dest'], 
                fl['dep_time'], 
                fl['arr_time'], 
                fl['pax'], 
                fl['fare']
            ] for fl in flights_data
        ], dtype=np.float32)
        
        self.flight_assigned = np.zeros(self.num_flights, dtype=bool)
        self.assignment_history = []
        return self._get_obs(), self._get_masks()

    def _get_obs(self):
        virtual_flight = np.zeros((1, 6), dtype=np.float32) 
        active_flights = self.flights[~self.flight_assigned]
        
        obs_flights = np.vstack([active_flights, virtual_flight]) if len(active_flights) > 0 else virtual_flight
        
        # Dimensionnement à max_flights + 1 pour inclure de façon permanente l'action de spill/wait
        pad_len = (self.max_flights + 1) - len(obs_flights)
        pad_mask = np.zeros(self.max_flights + 1, dtype=bool)
        
        if pad_len > 0:
            obs_flights = np.vstack([obs_flights, np.zeros((pad_len, 6), dtype=np.float32)])
            pad_mask[-pad_len:] = True
            
        return torch.tensor(self.agents), torch.tensor(obs_flights), torch.tensor(pad_mask)

    def _get_masks(self):
        active_flights = self.flights[~self.flight_assigned]
        num_active = len(active_flights)
        
        # Matrice augmentée (+1) pour prévenir le dépassement d'index à l'initialisation
        masks = np.zeros((self.num_agents, self.max_flights + 1), dtype=bool)
        
        for i in range(self.num_agents):
            pos = self.agents[i, 0]
            dispo = self.agents[i, 1]
            
            for j in range(num_active):
                origin = active_flights[j, 0]
                dep_time = active_flights[j, 2]
                if pos == origin and dispo <= dep_time:
                    masks[i, j] = True
            
            # Le vol virtuel est toujours positionné juste après les vols actifs
            masks[i, num_active] = True 

        return torch.tensor(masks)

    def step(self, actions):
        actions = actions.cpu().numpy()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        active_indices = np.where(~self.flight_assigned)[0]
        num_active = len(active_indices)
        
        target_dict = {}
        for i, act in enumerate(actions):
            if act < num_active:
                flight_idx = active_indices[act]
                if flight_idx not in target_dict:
                    target_dict[flight_idx] = []
                target_dict[flight_idx].append(i)
            else:
                self.agents[i, 1] += 60.0 

        for f_idx, agents_targeting in target_dict.items():
            best_agent = -1
            best_margin = -float('inf')
            
            for a_idx in agents_targeting:
                cap = self.agents[a_idx, 2]
                cost = self.agents[a_idx, 4]
                pax = min(cap, self.flights[f_idx, 4])
                fare = self.flights[f_idx, 5]
                
                margin = (pax * fare) - cost
                if margin > best_margin:
                    best_margin = margin
                    best_agent = a_idx
            
            rewards[best_agent] = best_margin
            self.flight_assigned[f_idx] = True

            self.assignment_history.append({
                'flight_index': f_idx,
                'agent_index': best_agent,
                'margin': best_margin
            })
            
            self.agents[best_agent, 0] = self.flights[f_idx, 1] 
            flight_duration = self.flights[f_idx, 3] - self.flights[f_idx, 2]
            self.agents[best_agent, 1] = self.flights[f_idx, 2] + flight_duration + 50.0 

        done = self.flight_assigned.all()
        
        if done:
            unmet_pax = self.flights[~self.flight_assigned, 4]
            fares = self.flights[~self.flight_assigned, 5]
            spill_penalty = np.sum(unmet_pax * fares) * self.spill_penalty_coef
            rewards -= spill_penalty / self.num_agents

        obs_a, obs_f, pad_mask = self._get_obs()
        masks = self._get_masks()
        
        return obs_a, obs_f, pad_mask, masks, torch.tensor(rewards), done