import logging

import geopandas as gp
import pandas as pd

from intro_matsim import gis
from intro_matsim.const import SUBTOUR_ID

logger = logging.getLogger(__name__)


def read_layer(gpkg_path, layer_name) -> gp.GeoDataFrame:
    """Read generic layer from a geopackage"""
    layer = gis.read_geopackage_layer(
        gpkg_path, layer_name=layer_name, target_crs=gis.AUSTRIA_LAMBERT
    )
    logger.info("Read %s items", len(layer))
    return layer


def read_car_network(gpkg_path, layer_name) -> gp.GeoDataFrame:
    """Return the car network with id as index"""
    network = gis.read_geopackage_layer(
        gpkg_path, layer_name=layer_name, target_crs=gis.AUSTRIA_LAMBERT
    )
    logger.info("Read full network with %s links", len(network))
    car_network = network[network["modes"].str.contains("car")].set_index("id")
    logger.info("Filtered car network with %s links", len(car_network))
    return car_network


def read_park_drive_stations(gpkg_path, layer_name) -> gp.GeoDataFrame:
    """Return the park and drive stations with name as index"""
    park_drive = gis.read_geopackage_layer(
        gpkg_path, layer_name=layer_name, target_crs=gis.AUSTRIA_LAMBERT
    )
    logger.info("Read %s park & drive stations", len(park_drive))
    return park_drive.set_index("name")


def read_cities(gpkg_path, layer_name) -> gp.GeoDataFrame:
    """Return cities with name as index"""
    cities = gis.read_geopackage_layer(
        gpkg_path, layer_name=layer_name, target_crs=gis.AUSTRIA_LAMBERT
    )
    logger.info("Read %s cities", len(cities))
    return cities.set_index("name")


def build_all_car_trips(
    trips: pd.DataFrame, car_network: gp.GeoDataFrame
) -> gp.GeoDataFrame:
    """
    Build a GeoDataFrame of all car trips with route geometries
    from the given trips (with link ids) and a network GeoDataFrame.
    Car trips without valid geometry are dropped.
    """
    if trips.empty or car_network.empty:
        return gp.GeoDataFrame()

    raw_trips = trips.loc[trips["mode"] == "car"]
    raw_subtour_count = raw_trips.groupby(SUBTOUR_ID).ngroups
    logger.info(
        "Got %s subtours consisting of %s car trips",
        raw_subtour_count,
        len(raw_trips),
    )

    trips = raw_trips[raw_trips.links.str.len() > 0]
    subtour_count = trips.groupby(SUBTOUR_ID).ngroups
    logger.info(
        "After removing trips without links: %s subtours consisting of %s car trips remain",
        subtour_count,
        len(trips),
    )
    logger.info("Merging link geometries. This takes a while..")
    # SLOW: takes ~7 minutes
    geometry = trips.apply(
        lambda row: gis.build_trip_geometry_from_link_ids(row.links, car_network),
        axis=1,
    )
    gdf_trips = gp.GeoDataFrame(trips, geometry=geometry, crs=car_network.crs)
    logger.info("Extracted geometries for %s trips", len(trips))
    return gdf_trips


def read_trips(trips_gpkg_path, station_name: str | None = None):
    """Load trips from a geopackage, optionally only for a certain station"""
    where = f'"{station_name}"=true' if station_name else None
    return gis.read_geopackage_layer(
        trips_gpkg_path, where=where, target_crs=gis.AUSTRIA_LAMBERT
    )
