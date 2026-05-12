import pandas as pd
import numpy as np
import airportsdata
from datetime import datetime, timedelta
import random
from pathlib import Path

# --- PARAMÈTRES D'EXÉCUTION MODIFIABLES ---
HUB = "LFPO"
DESTINATIONS = [
    "LFMN", "LFBO", "LPPR", "LPPT", "LIRF", "LEMD", "LEBL", "LEZL", "LGIR",
    "EDDB", "LFTH", "LPFR", "LGAV", "LFBZ", "LEMG", "LFMP", "LEPA", "LFMT",
    "LICJ", "LFML", "ESSA", "LTFM", "LIPZ", "LEMH", "LIBD", "LIRN", "LGSR",
    "LGMK", "EKCH", "LOWW", "LGKR", "LEAL", "EIDW", "LICC", "LATI", "LPMA",
    "LGRP", "LIMC", "LEVC", "LIBR", "GCTS", "LGSA", "LGTS", "LMML", "LIEO",
    "LIRP", "LIEE", "GCRR", "GCLP", "LTAI", "LTBJ", "EGPH"
]

SPEED_KMH = 963
MIN_FLIGHT_TIME_MINS = 55
TURNAROUND_TIME_MINS = 45
START_HOUR = 6
END_HOUR = 23
AIRCRAFT_COUNT = 5

# Modélisation tarifaire : Fixe + (Distance * Coût kilométrique)
FARE_BASE = 50.0
FARE_PER_KM = 0.05

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "Flight_Schedule" / "schedule_fap.csv"

# ------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371 

def calculate_metrics(dist_km):
    time_mins = max(MIN_FLIGHT_TIME_MINS, (dist_km / SPEED_KMH) * 60)
    fare = round((FARE_BASE + (dist_km * FARE_PER_KM)) / 10) * 10
    return time_mins, int(fare)

def generate_schedule():
    airports = airportsdata.load('ICAO')
    hub_data = airports[HUB]
    
    # Pré-calcul des vecteurs distance/temps/tarif
    metrics = {}
    for dest in DESTINATIONS:
        if dest in airports:
            dist = haversine(hub_data['lon'], hub_data['lat'], airports[dest]['lon'], airports[dest]['lat'])
            time_mins, fare = calculate_metrics(dist)
            metrics[dest] = (time_mins, fare, dist) # Ajout de la distance en mémoire

    valid_destinations = list(metrics.keys())
    base_day = datetime(2024, 1, 1)
    schedule = []
    flight_number = 101

    for ac_id in range(1, AIRCRAFT_COUNT + 1):
        current_time = base_day.replace(hour=START_HOUR, minute=0)
        end_of_ops = base_day.replace(hour=END_HOUR, minute=0)
        
        while True:
            dest = random.choice(valid_destinations)
            duration_mins, fare, dist = metrics[dest]
            
            # Validation de la viabilité
            total_cycle_time = (duration_mins * 2) + TURNAROUND_TIME_MINS
            if current_time + timedelta(minutes=total_cycle_time) > end_of_ops:
                break
                
            # Segment Aller
            dep_time_out = current_time
            arr_time_out = dep_time_out + timedelta(minutes=duration_mins)
            
            schedule.append({
                "Flight#": f"TO{flight_number}",
                "From": HUB,
                "To": dest,
                "Dept Time": dep_time_out.strftime("%H%M"),
                "Arr Time": arr_time_out.strftime("%H%M"),
                "Tarif": fare,
                "Distance [km]": round(dist)
            })
            flight_number += 1
            current_time = arr_time_out + timedelta(minutes=TURNAROUND_TIME_MINS)
            
            # Segment Retour
            dep_time_in = current_time
            arr_time_in = dep_time_in + timedelta(minutes=duration_mins)
            
            schedule.append({
                "Flight#": f"TO{flight_number}",
                "From": dest,
                "To": HUB,
                "Dept Time": dep_time_in.strftime("%H%M"),
                "Arr Time": arr_time_in.strftime("%H%M"),
                "Tarif": fare,
                "Distance [km]": round(dist)
            })
            flight_number += 1
            current_time = arr_time_in + timedelta(minutes=TURNAROUND_TIME_MINS)

    df = pd.DataFrame(schedule)
    df.to_csv(OUTPUT_FILE, index=False, sep=",")
    print(f"Opération terminée. Matrice générée ({len(df)} vols) : {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_schedule()