import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FAPEnv(gym.Env):
    def __init__(self, schedule_df, fleet_df, num_airports, base_spill_cost=1.5):
        super(FAPEnv, self).__init__()
        
        self.schedule = schedule_df
        self.fleet = fleet_df.to_dict('records')
        self.num_aircraft = len(self.fleet)
        self.num_airports = num_airports
        self.base_spill_cost = base_spill_cost
        
        self.prob_aog = 0.005
        self.prob_weather = 0.02
        
        self.action_space = spaces.Discrete(self.num_aircraft + 1)
        obs_dim = 6 + (self.num_aircraft * 2) 
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
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
        
        # Sécurité : Fin de simulation
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
        
        # INITIALISATION DES MÉTRIQUES (Correction de l'UnboundLocalError)
        revenue = 0
        spill_cost = 0
        delay_minutes = 0
        
        if action == self.num_aircraft:
            # Cas : Aucun avion assigné (Spill total)
            spill_cost = flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost
            reward = - spill_cost
        else:
            ac = self.fleet_state[action]
            
            # VERROU PHYSIQUE : Rejet strict des actions hors-masque (téléportation/voyage temporel)
            if ac['position'] != flight['Origin_Idx'] or ac['available_time'] > flight['Dept_Time_Minutes']:
                spill_cost = flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost
                reward = - (spill_cost * 2.0) # Sur-pénalité pour violation physique
                # L'état de l'avion reste inchangé, la demande est perdue

            else:

                if np.random.rand() < self.prob_weather:
                    # Cas : Aléas météo
                    spill_cost = flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost * 1.5
                    reward = - spill_cost
                    ac['available_time'] = max(ac['available_time'], flight['Dept_Time_Minutes']) + 120 
                else:
                    # Cas : Vol opéré normalement
                    pax = min(ac['capacity'], flight['Predicted_Demand'])
                    revenue = pax * flight['Tarif']
                    spill_cost = (flight['Predicted_Demand'] - pax) * flight['Tarif'] * self.base_spill_cost
                    
                    delay_minutes = np.random.lognormal(mean=2.0, sigma=0.8) if np.random.rand() < 0.3 else 0
                    repair_time = np.random.exponential(scale=240) if np.random.rand() < self.prob_aog else 0
                    
                    delay_penalty = 0.5 * (delay_minutes ** 2) 
                    reward = revenue - ac['cost_per_flight'] - spill_cost - delay_penalty
                    
                    ac['position'] = flight['Dest_Idx']
                    flight_duration = flight['Arr_Time_Minutes'] - flight['Dept_Time_Minutes']
                    
                    base_time = max(ac['available_time'], flight['Dept_Time_Minutes'])
                    ac['available_time'] = base_time + flight_duration + ac['turnaround_time'] + delay_minutes + repair_time

        self.current_step += 1
        terminated = self.current_step >= len(self.schedule)
        
        # Transmission de la télémétrie via le dictionnaire info
        return self._get_obs(), reward / 100000.0, terminated, False, {
            'revenue': revenue,
            'spill_cost': spill_cost,
            'delay_minutes': delay_minutes
        }

    def _get_obs(self):
        if self.current_step >= len(self.schedule):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        flight = self.schedule.iloc[self.current_step]
        
        obs_flight = [
            (flight['Dept_Time_Minutes'] % 1440.0) / 1440.0, 
            (flight['Arr_Time_Minutes'] % 1440.0) / 1440.0,
            flight['Origin_Idx'] / self.num_airports,
            flight['Dest_Idx'] / self.num_airports,
            flight['Predicted_Demand'] / 300.0,
            flight['Tarif'] / 500.0
        ]
        
        obs_fleet = []
        for ac in self.fleet_state:
            # Delta temporel relatif stationnaire (écrêté à +/- 1 jour)
            time_delta = (ac['available_time'] - flight['Dept_Time_Minutes']) / 1440.0
            time_delta = np.clip(time_delta, -1.0, 1.0)
            
            obs_fleet.extend([
                ac['position'] / self.num_airports,
                time_delta
            ])
            
        return np.array(obs_flight + obs_fleet, dtype=np.float32)