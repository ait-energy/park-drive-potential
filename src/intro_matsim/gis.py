import logging

import geopandas as gp
import h3
import pandas as pd
import pyogrio
from pyproj import Transformer
from shapely import (
    Geometry,
    LineString,
    MultiLineString,
    Point,
    Polygon,
    buffer,
    get_parts,
    get_point,
    line_merge,
)
from shapely.ops import linemerge, split, transform

logger = logging.getLogger(__name__)

AUSTRIA_LAMBERT = "EPSG:3416"
WGS84 = "EPSG:4326"


def read_geopackage_layer(
    gpkg_path, layer_name=None, where=None, target_crs=None
) -> gp.GeoDataFrame:
    """
    Read a single layer from a geopackage file and optionally reproject it.
    If the geopackage contains only a single layer the name can be omitted.
    """
    layers = pyogrio.list_layers(gpkg_path)
    if len(layers) == 1:
        layer_name = layers[0][0]
    gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name, where=where)
    assert isinstance(gdf, gp.GeoDataFrame)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf


def read_geopackage(gpkg_path, target_crs=None) -> dict[str, gp.GeoDataFrame]:
    """
    Read all layers from a geopackage file and optionally reproject them
    """
    layers = pyogrio.list_layers(gpkg_path)
    gdfs = {}
    for layer in layers:
        layer_name = layer[0]
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name)
        assert isinstance(gdf, gp.GeoDataFrame)
        if target_crs is not None:
            gdf = gdf.to_crs(target_crs)
        gdfs[layer_name] = gdf
    return gdfs


def write_gpkg(gdf: gp.GeoDataFrame, filename, layer: str | None = None) -> None:
    """
    Save way to (over)write a GeoPackage layer (in WGS84 projection)

    Deals with the issue that GeoPackage export only allows one geometry col.
    We either have to drop the other cols or convert them e.g. to wkt.
    This method simply drops all non-active geometry columns
    """
    if gdf.empty:
        logger.warning(f"Skip writing of empty gdf to {filename}")
        return

    drop_candidates = [x for x in gdf.columns if x.startswith("geom")]
    if gdf.active_geometry_name is not None:
        drop_candidates.remove(gdf.active_geometry_name)

    gdf.drop(drop_candidates, axis=1).to_crs(WGS84).to_file(
        filename, layer=layer, driver="GPKG"
    )


def buffer_gdf(
    gdf: gp.GeoDataFrame,
    buffer_radius: float,
    barriers: gp.GeoDataFrame | None = None,
) -> gp.GeoDataFrame:
    """Buffer a GeoDataFrame, returning a copy of the original.
    Buffers are cut with barriers if given (largest remaining part is used)."""
    copy = gdf.copy()
    copy.geometry = gdf.buffer(buffer_radius)

    if barriers is None:
        return copy

    barrier_poly = barriers.union_all().buffer(100)

    def _keep_largest(geom):
        if geom.geom_type == "Polygon":
            return geom
        elif geom.geom_type == "MultiPolygon":
            return max(geom.geoms, key=lambda x: x.area)
        else:
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    copy.geometry = copy.geometry.apply(
        lambda geom: _keep_largest(geom.difference(barrier_poly))
    )
    return copy


def intersect_gdfs(gdf1: gp.GeoDataFrame, gdf2: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Return all items of gdf1 that intersect with gdf2"""
    return gdf1[gdf1.intersects(gdf2.union_all())]


def buffer_wgs84_point(
    point_wgs84: Point, buffer_radius: int, meter_based_crs
) -> Polygon:
    """
    Buffer a single point coordinate in WGS84 projection
    """
    transformer_to_lambert = Transformer.from_crs(
        WGS84, meter_based_crs, always_xy=True
    )
    transformer_to_wgs = Transformer.from_crs(meter_based_crs, WGS84, always_xy=True)

    point_lambert = transform(transformer_to_lambert.transform, point_wgs84)
    point_buffered_lambert = point_lambert.buffer(buffer_radius)
    point_buffered = transform(transformer_to_wgs.transform, point_buffered_lambert)
    return point_buffered


def build_trip_geometry_from_link_ids(
    link_ids: str, network: gp.GeoDataFrame
) -> Geometry:
    """
    Merge geometries for all given link ids into a single (Multi)LineString.

    Should usually return a LineString, but if the links are not consecutive
    a MultiLineString is given.
    Note, the returned geometries may be complex, i.e. include self intersections
    """
    route = network.loc[link_ids.split(",")]
    # merge / flatten to a multi-line string
    lines = []
    for geom in route.geometry:
        if isinstance(geom, LineString):
            lines.append(geom)
        elif isinstance(geom, MultiLineString):
            lines.extend(geom.geoms)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")
    multi_line = MultiLineString(lines)
    # merge as much as possible
    return line_merge(multi_line, directed=True)


def first_point(geom: Geometry) -> Point:
    """Can handle both MultiLineString and LineString"""
    return get_point(get_parts(geom)[0], 0)


def last_point(geom: Geometry) -> Point:
    """Can handle both MultiLineString and LineString"""
    return get_point(get_parts(geom)[-1], -1)


def point_to_h3cell(series: gp.GeoSeries, resolution: int) -> pd.Series:
    return series.to_crs(WGS84).apply(lambda p: h3.latlng_to_cell(p.y, p.x, resolution))


def h3cell_to_polygon(series: pd.Series) -> gp.GeoSeries:
    polys = series.apply(_h3cell_to_polygon)
    return gp.GeoSeries(polys, crs=WGS84)


def _h3cell_to_polygon(h3_cell) -> Polygon:
    boundary = h3.cell_to_boundary(h3_cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def add_geom_for_poly_split(
    trips: gp.GeoDataFrame, poly: gp.GeoDataFrame
) -> gp.GeoDataFrame:
    """
    Split trips by intersecting them with the given (union of) poly.
    Returns a copy of trip with new geometry columns:
    - geom_before: part before entering the polygon the first time
    - geom_middle: parts when the polygon is entered multiple time
    - geom_after: part after leafing the polygon the last time
    - geom_shared: same as geom_after if tripNr==1, geom_before otherwise

    The resulting geometries can be empty.
    In case trip and polygon do not intersect all columns are empty.

    Note: parts of the multi-line string must be in a sensible order,
    e.g. the first part must represent the trip start.

    Args:
        trips: GeoDataFrame with mandatory 'tripNr' column
    """
    if trips.crs is not None and trips.crs != poly.crs:
        poly = poly.to_crs(trips.crs)
    cutting_poly = poly.union_all()

    def _split(row):
        geoms_before = []
        geoms_middle = []
        geoms_after = []

        geom = row.geometry
        is_line = isinstance(geom, (LineString, MultiLineString))
        if is_line and geom.intersects(cutting_poly):
            split_result = split(geom, cutting_poly.boundary)
            was_inside = False

            for part in split_result.geoms:
                # using the midpoint instead of using an intersection
                # because that would lead to many false positives
                midpoint = part.interpolate(0.5, normalized=True)
                is_inside = cutting_poly.contains(midpoint)

                if is_inside:
                    geoms_middle.append(part)
                    geoms_middle.extend(geoms_after)
                    geoms_after = []  # clear on re-entering the polygon
                    was_inside = True
                else:  # is not inside
                    if was_inside:
                        geoms_after.append(part)
                    else:  # and was never inside
                        geoms_before.append(part)

        return dict(
            geom_before=linemerge(geoms_before),
            geom_middle=linemerge(geoms_middle),
            geom_after=linemerge(geoms_after),
        )

    new_cols = trips.apply(_split, axis=1, result_type="expand")
    for col in new_cols.columns:
        new_cols[col] = gp.GeoSeries(new_cols[col], crs=trips.crs)
    result = pd.concat([trips, new_cols], axis=1)
    result["geom_shared"] = result.apply(
        lambda x: x.geom_after if x.tripNr == 1 else x.geom_before, axis=1
    )
    result["geom_shared"] = gp.GeoSeries(result["geom_shared"], crs=trips.crs)
    return gp.GeoDataFrame(result, crs=trips.crs)


def clip_links(link_ids: str, geom: Geometry, network: gp.GeoDataFrame) -> str:
    """
    Given link ids (comma separated) and a geometry,
    return a (potentially) reduced link string only containing links
    that intersect the given (slightly buffered) geometry.
    """
    route = network.loc[link_ids.split(",")]
    buf_geom = buffer(geom, 1)
    clipped_links = route[route.intersects(buf_geom)]
    if clipped_links.empty:
        return ""
    return ",".join(clipped_links.index)
