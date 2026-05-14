import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FAPEnv(gym.Env):
    """
    Environnement stochastique modélisant le Fleet Assignment Problem.
    Implémente le masquage d'actions dynamique et l'injection d'aléas
    (retards log-normaux, pannes de Poisson, annulations de Bernoulli)
    pour forcer l'émergence d'une politique de couverture des risques.
    """
    def __init__(self, schedule_df, fleet_df, num_airports, base_spill_cost=1.5):
        super(FAPEnv, self).__init__()
        
        self.schedule = schedule_df
        self.fleet = fleet_df.to_dict('records')
        self.num_aircraft = len(self.fleet)
        self.num_airports = num_airports
        self.base_spill_cost = base_spill_cost
        
        # Paramétrage stochastique
        self.prob_aog = 0.005      # Probabilité de panne matérielle (AOG)
        self.prob_weather = 0.02   # Probabilité d'annulation météorologique exogène
        
        # Espace d'action : 0 à N-1 pour la flotte. N pour le refus/sous-traitance.
        self.action_space = spaces.Discrete(self.num_aircraft + 1)
        
        # Espace d'observation continu vectorisé.
        # Dimensions : Vol en cours (6) + État de chaque appareil (Position, Temps de disponibilité)
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
                'position': 0, # Index de l'aéroport Hub
                'available_time': 0 # Horizon temporel (minutes)
            })
            
        return self._get_obs(), {}

    def action_masks(self):
        """
        Détermine la matrice de légalité des actions.
        Bloque toute affectation d'un appareil hors position ou indisponible temporellement.
        """
        mask = np.zeros(self.num_aircraft + 1, dtype=np.int8)
        mask[-1] = 1 # L'action d'annulation est le parachute structurel, toujours autorisé.
        
        flight = self.schedule.iloc[self.current_step]
        flight_dept_time = flight['Dept_Time_Minutes']
        flight_origin = flight['Origin_Idx']
        
        for i, ac in enumerate(self.fleet_state):
            if ac['position'] == flight_origin and ac['available_time'] <= flight_dept_time:
                mask[i] = 1
        return mask

    def step(self, action):
        flight = self.schedule.iloc[self.current_step]
        reward = 0.0
        
        # Traitement de l'action : Refus
        if action == self.num_aircraft:
            reward = - (flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost)
            
        # Traitement de l'action : Affectation
        else:
            ac = self.fleet_state[action]
            
            # Injection aléa 1 : Annulation exogène
            is_weather_cancel = np.random.rand() < self.prob_weather
            
            if is_weather_cancel:
                reward = - (flight['Predicted_Demand'] * flight['Tarif'] * self.base_spill_cost * 1.5)
                # Blocage au sol, repousse le temps de disponibilité
                ac['available_time'] += 120 
            else:
                pax = min(ac['capacity'], flight['Predicted_Demand'])
                revenue = pax * flight['Tarif']
                spill_cost = (flight['Predicted_Demand'] - pax) * flight['Tarif'] * self.base_spill_cost
                
                # Injection aléa 2 : Retard opérationnel (Loi Log-Normale)
                delay_minutes = np.random.lognormal(mean=2.0, sigma=0.8) if np.random.rand() < 0.3 else 0
                
                # Injection aléa 3 : Panne matérielle (Loi Exponentielle)
                repair_time = 0
                if np.random.rand() < self.prob_aog:
                    repair_time = np.random.exponential(scale=240)
                
                # Pénalisation quadratique de la propagation du retard
                delay_penalty = 0.5 * (delay_minutes ** 2) 
                
                reward = revenue - ac['cost_per_flight'] - spill_cost - delay_penalty
                
                # Transition d'état : Déplacement spatial et avancement temporel
                ac['position'] = flight['Dest_Idx']
                flight_duration = flight['Arr_Time_Minutes'] - flight['Dept_Time_Minutes']
                
                ac['available_time'] = (flight['Dept_Time_Minutes'] + 
                                        flight_duration + 
                                        ac['turnaround_time'] + 
                                        delay_minutes + 
                                        repair_time)

        self.current_step += 1
        terminated = self.current_step >= len(self.schedule)
        
        return self._get_obs(), reward / 100000.0, terminated, False, {}

    def _get_obs(self):
        """
        Génère le vecteur d'état normalisé exigé par le réseau de neurones (PPO).
        """
        if self.current_step >= len(self.schedule):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        flight = self.schedule.iloc[self.current_step]
        
        obs_flight = [
            flight['Dept_Time_Minutes'] / 1440.0, 
            flight['Arr_Time_Minutes'] / 1440.0,
            flight['Origin_Idx'] / self.num_airports,
            flight['Dest_Idx'] / self.num_airports,
            flight['Predicted_Demand'] / 300.0, # Normalisation basée sur un MAX théorique
            flight['Tarif'] / 500.0
        ]
        
        obs_fleet = []
        for ac in self.fleet_state:
            obs_fleet.extend([
                ac['position'] / self.num_airports,
                ac['available_time'] / 1440.0 # Peut dépasser 1.0 (jours suivants)
            ])
            
        return np.array(obs_flight + obs_fleet, dtype=np.float32)