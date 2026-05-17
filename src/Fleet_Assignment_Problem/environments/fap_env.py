import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FAPEnv(gym.Env):
    def __init__(self, fleet_df, num_airports, base_spill_cost=1.5):
        super(FAPEnv, self).__init__()
        
        self.fleet = fleet_df.to_dict('records')
        self.num_aircraft = len(self.fleet)
        self.num_airports = num_airports
        self.base_spill_cost = base_spill_cost
        
        self.action_space = spaces.Discrete(self.num_aircraft + 1)
        
        # Dimensions One-Hot Encoding
        obs_flight_dim = 4 + (self.num_airports * 2) # 4 vars continues + OHE Origine + OHE Dest
        obs_fleet_dim = self.num_aircraft * (1 + self.num_airports) # Delta temps + OHE Position par avion
        obs_dim = obs_flight_dim + obs_fleet_dim
        
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Génération stochastique à chaque nouvel épisode
        self.schedule = build_network_state_for_episode(self.num_aircraft)
        
        self.fleet_state = []
        for ac in self.fleet:
            self.fleet_state.append({
                'capacity': ac['capacity'],
                'cost_per_flight': ac.get('cost_per_flight', 5000),
                'turnaround_time': ac.get('turnaround_time', 50),
                'position': 0, 
                'available_time': 0 
            })
            
        return self._get_obs(), {}

    def action_masks(self):
        mask = np.zeros(self.num_aircraft + 1, dtype=np.int8)
        mask[-1] = 1 
        
        if self.current_step >= len(self.schedule):
            return mask
            
        flight = self.schedule.iloc[self.current_step]
        for i, ac in enumerate(self.fleet_state):
            if ac['position'] == flight['Origin_Idx'] and ac['available_time'] <= flight['Dept_Time_Minutes']:
                mask[i] = 1
        return mask

    def step(self, action):
        flight = self.schedule.iloc[self.current_step]
        reward = 0.0
        revenue = 0
        spill_cost = 0
        
        if action == self.num_aircraft:
            spill_cost = flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost
            reward = - spill_cost
        else:
            ac = self.fleet_state[action]
            
            if ac['position'] != flight['Origin_Idx'] or ac['available_time'] > flight['Dept_Time_Minutes']:
                spill_cost = flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost
                reward = - (spill_cost * 2.0) 
            else:
                pax = min(ac['capacity'], flight['Predicted_Demand'])
                revenue = pax * flight['Tarif']
                spill_cost = (flight['Predicted_Demand'] - pax) * flight['Tarif'] * self.base_spill_cost
                
                reward = revenue - ac['cost_per_flight'] - spill_cost
                
                ac['position'] = flight['Dest_Idx']
                flight_duration = flight['Arr_Time_Minutes'] - flight['Dept_Time_Minutes']
                
                base_time = max(ac['available_time'], flight['Dept_Time_Minutes'])
                ac['available_time'] = base_time + flight_duration + ac['turnaround_time']

        self.current_step += 1
        terminated = self.current_step >= len(self.schedule)
        
        return self._get_obs(), reward / 100000.0, terminated, False, {
            'revenue': revenue,
            'spill_cost': spill_cost
        }

    def _get_obs(self):
        if self.current_step >= len(self.schedule):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        flight = self.schedule.iloc[self.current_step]
        
        # One-Hot Encoding Vol
        origin_ohe = np.zeros(self.num_airports, dtype=np.float32)
        origin_ohe[int(flight['Origin_Idx'])] = 1.0
        
        dest_ohe = np.zeros(self.num_airports, dtype=np.float32)
        dest_ohe[int(flight['Dest_Idx'])] = 1.0
        
        obs_flight = [
            (flight['Dept_Time_Minutes'] % 1440.0) / 1440.0, 
            (flight['Arr_Time_Minutes'] % 1440.0) / 1440.0,
            flight['Predicted_Demand'] / 300.0,
            flight['Tarif'] / 500.0
        ]
        obs_flight.extend(origin_ohe)
        obs_flight.extend(dest_ohe)
        
        # One-Hot Encoding Flotte
        obs_fleet = []
        for ac in self.fleet_state:
            time_delta = (ac['available_time'] - flight['Dept_Time_Minutes']) / 1440.0
            time_delta = np.clip(time_delta, -1.0, 1.0)
            
            pos_ohe = np.zeros(self.num_airports, dtype=np.float32)
            pos_ohe[int(ac['position'])] = 1.0
            
            obs_fleet.append(time_delta)
            obs_fleet.extend(pos_ohe)
            
        return np.array(obs_flight + obs_fleet, dtype=np.float32)