import geopandas as gp
from shapely import LineString, MultiLineString, Polygon

from intro_matsim.gis import WGS84, add_geom_for_poly_split, write_gpkg

SQUARE_POLYGON = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
SQUARE_POLYGON_GDF = gp.GeoDataFrame({"geometry": [SQUARE_POLYGON]}, crs=WGS84)
LINESTRING_EMPTY = LineString([])

# set path to a geopackage for easier visual debugging
GPKG = None  # "/tmp/intro_matsim_gis_test.gpkg"


def test_add_geom_for_poly_split__simple_line_completely_outside_square_polygon():
    line = LineString([(-1, -1), (-1, -2), (-1, -3)])
    trips = gp.GeoDataFrame({"tripNr": [1], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    result = result.iloc[0]

    assert result["geom_before"].equals(LINESTRING_EMPTY)
    assert result["geom_middle"].equals(LINESTRING_EMPTY)
    assert result["geom_after"].equals(LINESTRING_EMPTY)
    assert result["geom_shared"].equals(LINESTRING_EMPTY)


def test_add_geom_for_poly_split__simple_line_inside_square_polygon():
    line = LineString([(0.2, 0.2), (0.8, 0.8)])
    trips = gp.GeoDataFrame({"tripNr": [1], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    result = result.iloc[0]

    middle = LineString([(0.2, 0.2), (0.8, 0.8)])
    assert result["geom_before"].equals(LINESTRING_EMPTY)
    assert result["geom_middle"].equals(middle)
    assert result["geom_after"].equals(LINESTRING_EMPTY)
    assert result["geom_shared"].equals(LINESTRING_EMPTY)


def test_add_geom_for_poly_split__simple_line_crossing_square_polygon(request):
    line = LineString([(-1, 0.5), (0.5, 0.5), (2, 0.5)])
    trips = gp.GeoDataFrame({"tripNr": [1], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    _write_debugging_gpkg(request, trips, SQUARE_POLYGON_GDF, result)
    result = result.iloc[0]

    before = LineString([(-1, 0.5), (0, 0.5)])
    middle = LineString([(0, 0.5), (1, 0.5)])
    after = LineString([(1, 0.5), (2, 0.5)])
    assert result["geom_before"].equals(before)
    assert result["geom_middle"].equals(middle)
    assert result["geom_after"].equals(after)
    assert result["geom_shared"].equals(after)


def test_add_geom_for_poly_split__simple_line_crossing_square_polygon_tripNr0(request):
    line = LineString([(-1, 0.5), (0.5, 0.5), (2, 0.5)])
    trips = gp.GeoDataFrame({"tripNr": [0], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    _write_debugging_gpkg(request, trips, SQUARE_POLYGON_GDF, result)
    result = result.iloc[0]

    before = LineString([(-1, 0.5), (0, 0.5)])
    middle = LineString([(0, 0.5), (1, 0.5)])
    after = LineString([(1, 0.5), (2, 0.5)])
    assert result["geom_before"].equals(before)
    assert result["geom_middle"].equals(middle)
    assert result["geom_after"].equals(after)
    assert result["geom_shared"].equals(before)


def test_add_geom_for_poly_split__line_with_self_intersection_at_beginning(request):
    line = LineString(
        [
            (-3, 0.5),
            (-2, 0.5),
            (-2.5, 1),
            (-2, 1),
            (-2, 0.5),
            (-1, 0.5),
            (0.5, 0.5),
            (2, 0.5),
        ]
    )
    trips = gp.GeoDataFrame({"tripNr": [1], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    _write_debugging_gpkg(request, trips, SQUARE_POLYGON_GDF, result)
    result = result.iloc[0]

    before = LineString(
        [(-3, 0.5), (-2, 0.5), (-2.5, 1), (-2, 1), (-2, 0.5), (-1, 0.5), (0, 0.5)]
    )
    middle = LineString([(0, 0.5), (1, 0.5)])
    after = LineString([(1, 0.5), (2, 0.5)])
    assert result["geom_before"].equals(before)
    assert result["geom_middle"].equals(middle)
    assert result["geom_after"].equals(after)
    assert result["geom_shared"].equals(after)


def test_add_geom_for_poly_split__multiline_with_self_intersection_at_beginning(
    request,
):
    line = MultiLineString(
        [
            [(-3, 0.5), (-2, 0.5)],
            [(-2, 0.5), (-2.5, 1), (-2, 1), (-2, 0.5)],
            [(-2, 0.5), (-1, 0.5), (0.5, 0.5), (2, 0.5)],
        ]
    )
    trips = gp.GeoDataFrame({"tripNr": [1], "geometry": [line]}, crs=WGS84)

    result = add_geom_for_poly_split(trips, SQUARE_POLYGON_GDF)
    assert isinstance(result, gp.GeoDataFrame)
    _write_debugging_gpkg(request, trips, SQUARE_POLYGON_GDF, result)
    result = result.iloc[0]

    before = LineString(
        [(-3, 0.5), (-2, 0.5), (-2.5, 1), (-2, 1), (-2, 0.5), (-1, 0.5), (0, 0.5)]
    )
    middle = LineString([(0, 0.5), (1, 0.5)])
    after = LineString([(1, 0.5), (2, 0.5)])
    assert result["geom_before"].equals(before)
    assert result["geom_middle"].equals(middle)
    assert result["geom_after"].equals(after)
    assert result["geom_shared"].equals(after)


def _write_debugging_gpkg(request, trips, poly, result):
    if not GPKG:
        return

    name = request.node.name.replace("test_add_geom_for_poly_split__", "")
    write_gpkg(trips, GPKG, layer=f"{name}_trips")
    write_gpkg(poly, GPKG, layer=f"{name}_poly")
    write_gpkg(result.set_geometry("geom_before"), GPKG, layer=f"{name}_before")
    write_gpkg(result.set_geometry("geom_middle"), GPKG, layer=f"{name}_middle")
    write_gpkg(result.set_geometry("geom_after"), GPKG, layer=f"{name}_after")
    write_gpkg(result.set_geometry("geom_shared"), GPKG, layer=f"{name}_shared")
