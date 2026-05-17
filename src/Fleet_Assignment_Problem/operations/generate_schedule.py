import pandas as pd
import numpy as np
import airportsdata
from datetime import datetime, timedelta
import random

def generate_dynamic_schedule(num_aircraft, days_to_simulate=7, base_date=datetime(2025, 1, 1)):
    HUB = "LFPO"
    DESTINATIONS = ["LFMN", "LEMD", "LPPT", "LIRF", "EDDB", "GCTS", "LGAV", "EIDW", "GCLP", "LOWW"]
    SPEED_KMH = 850
    TURNAROUND_TIME_MINS = 50
    OPERATING_HOURS = {"start": 6, "end": 22}
    FARE_BASE = 50.0
    FARE_PER_KM = 0.05

    airports = airportsdata.load('ICAO')

    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        return 6371 * (2 * np.arcsin(np.sqrt(a)))

    distances = {dest: haversine(airports[HUB]['lon'], airports[HUB]['lat'], airports[dest]['lon'], airports[dest]['lat']) for dest in DESTINATIONS}

    schedule = []
    flight_id = 1000

    for ac_idx in range(num_aircraft):
        current_time = base_date.replace(hour=OPERATING_HOURS["start"], minute=random.choice([0, 15, 30, 45]))
        current_time += timedelta(minutes=random.randint(0, 90))
        end_date = base_date + timedelta(days=days_to_simulate)

        while current_time < end_date:
            if current_time.hour >= OPERATING_HOURS["end"] or current_time.hour < OPERATING_HOURS["start"]:
                current_time = (current_time + timedelta(days=1)).replace(hour=OPERATING_HOURS["start"], minute=random.choice([0, 15, 30]))
                if current_time >= end_date: break

            dest = random.choice(DESTINATIONS)
            dist = distances[dest]
            flight_duration = timedelta(minutes=int(np.ceil((dist / SPEED_KMH) * 60)))
            fare = int(FARE_BASE + (dist * FARE_PER_KM))

            if (current_time + (flight_duration * 2) + timedelta(minutes=TURNAROUND_TIME_MINS)).day != current_time.day:
                current_time = (current_time + timedelta(days=1)).replace(hour=OPERATING_HOURS["start"], minute=0)
                continue

            arr_time_out = current_time + flight_duration
            schedule.append({"Flight#": f"TO{flight_id}", "From": HUB, "To": dest, "Dept Time": current_time, "Arr Time": arr_time_out, "Distance [km]": round(dist), "Tarif": fare})
            flight_id += 1

            dep_time_in = arr_time_out + timedelta(minutes=TURNAROUND_TIME_MINS)
            arr_time_in = dep_time_in + flight_duration
            schedule.append({"Flight#": f"TO{flight_id}", "From": dest, "To": HUB, "Dept Time": dep_time_in, "Arr Time": arr_time_in, "Distance [km]": round(dist), "Tarif": fare})
            flight_id += 1

            current_time = arr_time_in + timedelta(minutes=TURNAROUND_TIME_MINS + random.choice([0, 30, 60]))

    return pd.DataFrame(schedule).sort_values(by="Dept Time").reset_index(drop=True)