# Merging GPX (GPS) + TCX (time, HR, other metrics) correctly
# Strategy:
# - Keep GPS coordinates exactly as they are in Afternoon_Run.gpx
# - Map TCX time and numeric metrics (HeartRate, Cadence, Watts/Power, Speed if present) onto GPX points
#   by using proportional progress along the GPX route:
#     fraction_along_gpx = distance_along_point / total_gpx_distance
#     target_time = tcx_start_time + fraction_along_gpx * tcx_total_duration
# - Interpolate numeric metrics from TCX using time
#
# Output: merged_output_corrected.tcx (Strava-friendly TCX containing GPX positions + TCX metrics)

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import math
import os

tcx_path = "/mnt/data/Zepp20250809170205-2.tcx"
gpx_path = "/mnt/data/Afternoon_Run.gpx"
output_path = "/mnt/data/merged_output_corrected.tcx"

def localname(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def parse_time(tstr):
    if tstr is None:
        return None
    tstr = tstr.strip()
    # Handle trailing Z or timezone offsets
    if tstr.endswith('Z'):
        return datetime.fromisoformat(tstr.replace('Z', '+00:00')).astimezone(timezone.utc)
    try:
        return datetime.fromisoformat(tstr).astimezone(timezone.utc)
    except Exception:
        # try without timezone
        try:
            return datetime.fromisoformat(tstr).replace(tzinfo=timezone.utc)
        except Exception:
            return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

# --- Parse GPX (keep GPS points exactly) ---
gpx_tree = ET.parse(gpx_path)
gpx_root = gpx_tree.getroot()

gpx_points = []
for el in gpx_root.iter():
    if localname(el.tag) == 'trkpt':
        lat = float(el.attrib.get('lat'))
        lon = float(el.attrib.get('lon'))
        ele = None
        for c in el:
            if localname(c.tag) == 'ele':
                try:
                    ele = float(c.text)
                except:
                    ele = None
        gpx_points.append({'lat': lat, 'lon': lon, 'ele': ele, 'xml': el})

if len(gpx_points) == 0:
    raise SystemExit("No GPX trackpoints found in Afternoon_Run.gpx")

# compute cumulative distances along GPX
cumdist = [0.0]
for i in range(1, len(gpx_points)):
    p0 = gpx_points[i-1]
    p1 = gpx_points[i]
    d = haversine(p0['lat'], p0['lon'], p1['lat'], p1['lon'])
    cumdist.append(cumdist[-1] + d)

total_gpx_distance = cumdist[-1]

# --- Parse TCX (extract times and numeric metrics) ---
tcx_tree = ET.parse(tcx_path)
tcx_root = tcx_tree.getroot()

tcx_tps = []
for el in tcx_root.iter():
    if localname(el.tag) == 'Trackpoint':
        tp = el
        # find Time child
        t_el = None
        for c in tp:
            if localname(c.tag) == 'Time':
                t_el = c
                break
        t = parse_time(t_el.text) if t_el is not None else None

        # heart rate
        hr = None
        for c in tp:
            if localname(c.tag) == 'HeartRateBpm':
                for cc in c:
                    if localname(cc.tag) == 'Value':
                        try:
                            hr = int(cc.text)
                        except:
                            hr = None
                        break
                break

        # cadence
        cadence = None
        for c in tp:
            if localname(c.tag) == 'Cadence':
                try:
                    cadence = int(c.text)
                except:
                    cadence = None
                break

        # distanceMeters in TCX (sometimes present)
        dist = None
        for c in tp:
            if localname(c.tag) == 'DistanceMeters':
                try:
                    dist = float(c.text)
                except:
                    dist = None
                break

        # power/watts & speed often inside Extensions -> TPX
        watts = None
        speed = None
        for c in tp.iter():
            lname = localname(c.tag)
            if lname in ('Watts', 'Power'):
                try:
                    watts = int(c.text)
                except:
                    watts = None
            if lname == 'Speed':
                try:
                    speed = float(c.text)
                except:
                    speed = None

        tcx_tps.append({'time': t, 'hr': hr, 'cadence': cadence, 'dist': dist, 'watts': watts, 'speed': speed})

# filter out trackpoints without time
tcx_tps = [tp for tp in tcx_tps if tp['time'] is not None]
if len(tcx_tps) == 0:
    raise SystemExit("No timestamped Trackpoints found in Zepp TCX file. Cannot map metrics without time data.")

tcx_start = tcx_tps[0]['time']
tcx_end = tcx_tps[-1]['time']
tcx_total_seconds = (tcx_end - tcx_start).total_seconds()
if tcx_total_seconds <= 0:
    raise SystemExit("TCX time span is zero or negative; cannot map.")

# build time/value arrays (seconds from start)
tcx_seconds = [(tp['time'] - tcx_start).total_seconds() for tp in tcx_tps]
tcx_hr_vals = [tp['hr'] for tp in tcx_tps]
tcx_cadence_vals = [tp['cadence'] for tp in tcx_tps]
tcx_watts_vals = [tp['watts'] for tp in tcx_tps]
tcx_speed_vals = [tp['speed'] for tp in tcx_tps]

# helper for linear interpolation of numeric metric vs time
def interp_metric(t_seconds, times_array, values_array):
    # returns None if no numeric values available at all
    numeric_indices = [i for i, v in enumerate(values_array) if v is not None]
    if not numeric_indices:
        return None
    # if t is before first time or after last time, clamp to nearest available numeric value
    if t_seconds <= times_array[0]:
        # first numeric index
        first_idx = numeric_indices[0]
        return values_array[first_idx]
    if t_seconds >= times_array[-1]:
        last_idx = numeric_indices[-1]
        return values_array[last_idx]
    # find bounding indices
    for i in range(len(times_array)-1):
        t0, t1 = times_array[i], times_array[i+1]
        if t0 <= t_seconds <= t1:
            v0, v1 = values_array[i], values_array[i+1]
            # if both present -> linear interpolation
            if v0 is not None and v1 is not None:
                if isinstance(v0, int) and isinstance(v1, int):
                    # return integer interpolation (rounded)
                    return int(round(v0 + (v1 - v0) * (t_seconds - t0) / (t1 - t0)))
                else:
                    return v0 + (v1 - v0) * (t_seconds - t0) / (t1 - t0)
            # fallback: find nearest numeric neighbor
            # search outward for nearest numeric value
            left = i
            right = i+1
            # move left
            while left >= 0 and values_array[left] is None:
                left -= 1
            while right < len(values_array) and values_array[right] is None:
                right += 1
            candidates = []
            if left >= 0:
                candidates.append((abs(t_seconds - times_array[left]), values_array[left]))
            if right < len(values_array):
                candidates.append((abs(times_array[right] - t_seconds), values_array[right]))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
            return None
    return None

# --- Map GPX points to TCX times & metrics ---
mapped = []
n_points = len(gpx_points)
for i, gp in enumerate(gpx_points):
    if total_gpx_distance > 0.5:  # if route has measurable distance, use proportional mapping
        fraction = cumdist[i] / total_gpx_distance
    else:
        # fallback: map by index proportion
        fraction = i / max(1, (n_points-1))
    tgt_seconds = fraction * tcx_total_seconds
    tgt_time = tcx_start + timedelta(seconds=tgt_seconds)
    # interpolated metrics
    hr_v = interp_metric(tgt_seconds, tcx_seconds, tcx_hr_vals)
    cad_v = interp_metric(tgt_seconds, tcx_seconds, tcx_cadence_vals)
    watts_v = interp_metric(tgt_seconds, tcx_seconds, tcx_watts_vals)
    speed_v = interp_metric(tgt_seconds, tcx_seconds, tcx_speed_vals)
    mapped.append({'time': tgt_time, 'lat': gp['lat'], 'lon': gp['lon'], 'ele': gp['ele'],
                   'hr': hr_v, 'cadence': cad_v, 'watts': watts_v, 'speed': speed_v})

# --- Build TCX with GPX positions + TCX metrics ---
ns = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'
root = ET.Element('TrainingCenterDatabase', {
    'xmlns': ns,
    'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'xsi:schemaLocation': ns + ' http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd'
})

activities = ET.SubElement(root, 'Activities')
activity = ET.SubElement(activities, 'Activity', Sport="Running")
ET.SubElement(activity, 'Id').text = tcx_start.strftime('%Y-%m-%dT%H:%M:%SZ')

lap = ET.SubElement(activity, 'Lap', StartTime=tcx_start.strftime('%Y-%m-%dT%H:%M:%SZ'))
ET.SubElement(lap, 'TotalTimeSeconds').text = str(int(tcx_total_seconds))
ET.SubElement(lap, 'DistanceMeters').text = str(int(round(total_gpx_distance)))  # from GPX
ET.SubElement(lap, 'Calories').text = "0"
ET.SubElement(lap, 'Intensity').text = "Active"
ET.SubElement(lap, 'TriggerMethod').text = "Manual"

track = ET.SubElement(lap, 'Track')
for pt in mapped:
    tp = ET.SubElement(track, 'Trackpoint')
    ET.SubElement(tp, 'Time').text = pt['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
    pos = ET.SubElement(tp, 'Position')
    ET.SubElement(pos, 'LatitudeDegrees').text = f"{pt['lat']:.8f}"
    ET.SubElement(pos, 'LongitudeDegrees').text = f"{pt['lon']:.8f}"
    if pt['ele'] is not None:
        ET.SubElement(tp, 'AltitudeMeters').text = f"{pt['ele']}"
    if pt['hr'] is not None:
        hr_el = ET.SubElement(tp, 'HeartRateBpm')
        ET.SubElement(hr_el, 'Value').text = str(int(round(pt['hr'])))
    if pt['cadence'] is not None:
        ET.SubElement(tp, 'Cadence').text = str(int(round(pt['cadence'])))
    # add Extensions/TPX for speed/watts if present
    if pt['speed'] is not None or pt['watts'] is not None:
        ext = ET.SubElement(tp, 'Extensions')
        tpx = ET.SubElement(ext, 'TPX')  # not adding Garmin namespace to keep simple
        if pt['speed'] is not None:
            ET.SubElement(tpx, 'Speed').text = str(float(pt['speed']))
        if pt['watts'] is not None:
            ET.SubElement(tpx, 'Watts').text = str(int(round(pt['watts'])))

# Save file
tree = ET.ElementTree(root)
tree.write(output_path, encoding='UTF-8', xml_declaration=True)

# Summary printout for the user
summary = {
    'gpx_points': len(gpx_points),
    'total_gpx_distance_m': round(total_gpx_distance, 1),
    'tcx_start_utc': tcx_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'tcx_end_utc': tcx_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'tcx_duration_s': int(tcx_total_seconds),
    'output_path': output_path,
    'file_exists': os.path.exists(output_path)
}

summary, mapped[:3]  # return summary and first 3 mapped points as preview

