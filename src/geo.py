"""Approximate coordinates for metro areas (location of the first principal city).

Uses the offline GeoNames city list bundled in the `geonamescache` package, so no download is
needed. Nonmetropolitan areas are regions without a single centre and are not placed on the map.
"""
import re

import geonamescache
import pandas as pd

# principal cities missing from the GeoNames list (coordinates of the city centre)
MANUAL = {
    "Urban Honolulu, HI": (21.307, -157.858),
    "Winston-Salem, NC": (36.100, -80.244),
    "Ponce, PR": (18.011, -66.614),
    "Arecibo, PR": (18.472, -66.716),
    "Mayaguez, PR": (18.201, -67.139),
    "Guayama, PR": (17.984, -66.113),
}


def _city_lookup():
    lut = {}
    for c in geonamescache.GeonamesCache(min_city_population=500).get_cities().values():
        if c["countrycode"] != "US":
            continue
        key = (c["name"].lower(), c["admin1code"])
        if key not in lut or c["population"] > lut[key][2]:
            lut[key] = (c["latitude"], c["longitude"], c["population"])
    return lut


def metro_coordinates(areas: pd.DataFrame) -> pd.DataFrame:
    """areas: columns AREA, AREA_TITLE -> AREA, lat, lon (metro areas only)."""
    lut = _city_lookup()
    rows = []
    for area, title in areas[["AREA", "AREA_TITLE"]].drop_duplicates().itertuples(index=False):
        if "nonmetropolitan" in title.lower() or ", " not in title:
            continue
        if title in MANUAL:
            lat, lon = MANUAL[title]
        else:
            city, state = title.rsplit(", ", 1)
            city, state = city.split("-")[0].split("/")[0].strip(), state.split("-")[0]
            cands = [city, city.replace("St.", "Saint"), city.replace("Saint", "St."),
                     re.sub(r" (Town|City)$", "", city), city + " City"]
            hit = next((lut[(c.lower(), state)] for c in cands if (c.lower(), state) in lut), None)
            if hit is None:
                continue
            lat, lon = hit[0], hit[1]
        rows.append({"AREA": area, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)
