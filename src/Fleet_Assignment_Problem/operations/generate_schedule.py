import pandas as pd
import numpy as np
import airportsdata
from datetime import datetime, timedelta
import random
from pathlib import Path

# Paramètres de simulation
HUB = "LFPO"
DESTINATIONS = ["LFMN", "LEMD", "LPPT", "LIRF", "EDDB", "GCTS", "LGAV", "EIDW", "GCLP", "LOWW"]
SPEED_KMH = 850
TURNAROUND_TIME_MINS = 50
DAYS_TO_SIMULATE = 7
BASE_DATE = datetime(2025, 1, 1)

# Paramètres tarifaires
FARE_BASE = 50.0
FARE_PER_KM = 0.05

# Vagues de départs (Hub Banks)
BANKS = [
    {"start": 6, "end": 8, "flights": 8},   # Vague matinale
    {"start": 12, "end": 14, "flights": 6}, # Vague méridienne
    {"start": 18, "end": 20, "flights": 8}  # Vague du soir
]

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371

airports = airportsdata.load('ICAO')
hub_lat, hub_lon = airports[HUB]['lat'], airports[HUB]['lon']

schedule = []
flight_id = 1000

for day in range(DAYS_TO_SIMULATE):
    current_date = BASE_DATE + timedelta(days=day)
    
    for bank in BANKS:
        wave_dests = random.sample(DESTINATIONS, min(bank["flights"], len(DESTINATIONS)))
        
        for dest in wave_dests:
            dest_lat, dest_lon = airports[dest]['lat'], airports[dest]['lon']
            dist = haversine(hub_lon, hub_lat, dest_lon, dest_lat)
            flight_duration = timedelta(minutes=int((dist / SPEED_KMH) * 60))
            
            # Calcolo della tariffa
            fare = round((FARE_BASE + (dist * FARE_PER_KM)) / 10) * 10
            
            dep_hour = random.randint(bank["start"], bank["end"] - 1)
            dep_minute = random.choice([0, 10, 15, 20, 30, 40, 45, 50])
            dep_time_out = current_date.replace(hour=dep_hour, minute=dep_minute)
            arr_time_out = dep_time_out + flight_duration
            
            # Segment Aller
            schedule.append({
                "Flight#": f"TO{flight_id}",
                "From": HUB,
                "To": dest,
                "Dept Time": dep_time_out,
                "Arr Time": arr_time_out,
                "Distance [km]": round(dist),
                "Tarif": fare
            })
            flight_id += 1
            
            # Segment Retour
            dep_time_in = arr_time_out + timedelta(minutes=TURNAROUND_TIME_MINS)
            arr_time_in = dep_time_in + flight_duration
            
            schedule.append({
                "Flight#": f"TO{flight_id}",
                "From": dest,
                "To": HUB,
                "Dept Time": dep_time_in,
                "Arr Time": arr_time_in,
                "Distance [km]": round(dist),
                "Tarif": fare
            })
            flight_id += 1

df_schedule = pd.DataFrame(schedule)
df_schedule = df_schedule.sort_values(by="Dept Time").reset_index(drop=True)

BASE_DIR = Path(__file__).resolve().parents[3]
# Creazione delle directory se mancanti
output_dir = BASE_DIR / "data" / "raw" / "Flight_Schedule"
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = output_dir / "schedule_fap.csv"
df_schedule.to_csv(OUTPUT_FILE, index=False)