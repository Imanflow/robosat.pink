import argparse
import collections
import struct
import json
import sys
import os
import io

import numpy as np
from PIL import Image
from tqdm import tqdm

import mercantile
import fiona
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles

from robosat_pink.config import load_config
from robosat_pink.colors import make_palette, complementary_palette
from robosat_pink.tiles import tiles_from_csv
from robosat_pink.web_ui import web_ui
from robosat_pink.logs import Logs

import psycopg2


def add_parser(subparser):
    parser = subparser.add_parser(
        "rasterize",
        help="rasterize either GeoJSON or PostGIS features to raster labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="path to configuration file")
    parser.add_argument("--postgis", type=str, help="PostGIS SQL SELECT query to retrieve features")
    parser.add_argument("--geojson", type=str, nargs="+", help="path to GeoJSON features files")
    parser.add_argument("--zoom", type=int, required="--geojson" in sys.argv, help="zoom level of tiles (for GeoJSON)")
    parser.add_argument("--tile_size", type=int, help="if set, override tile size value from config file")
    parser.add_argument("--web_ui", action="store_true", help="activate web ui output")
    parser.add_argument("--web_ui_base_url", type=str, help="web ui alternate base url")
    parser.add_argument("--web_ui_template", type=str, help="path to an alternate web ui template")
    parser.add_argument("cover", type=str, help="path to csv tiles cover file")
    parser.add_argument("out", type=str, help="directory to write converted images")

    parser.set_defaults(func=main)


def geojson_to_mercator(feature, epsg=4326):
    """Convert GeoJSON Polygon feature coords to Mercator (i.e EPSG:3857).
       Inspired by: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe
    """

    if feature["geometry"]["type"] == "Polygon":
        xys = (zip(*ring) for ring in feature["geometry"]["coordinates"])
        xys = (list(zip(*transform(CRS.from_epsg(int(epsg)), CRS.from_epsg(3857), *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}


def geojson_tile_burn(tile, features, tile_size, burn_value=1, epsg=4326):
    """Burn tile with GeoJSON features."""

    shapes = ((geometry, burn_value) for feature in features for geometry in geojson_to_mercator(feature, epsg))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, tile_size, tile_size)

    return rasterize(shapes, out_shape=(tile_size, tile_size), transform=transform)


def wkb_to_numpy(wkb):
    """Convert a PostGIS WKB raster to a NumPy array.
       Inspired by: https://github.com/nathancahill/wkb-raster

       PostGIS WKB RFC: http://trac.osgeo.org/postgis/browser/trunk/raster/doc/RFC2-WellKnownBinaryFormat
    """

    out = None

    if not wkb:
        return None

    endian = ">" if struct.unpack("<b", wkb.read(1)) == 0 else "<"  # raster Endiannes
    (_, bands, _, _, _, _, _, _, srid, width, height) = struct.unpack(
        endian + "HHddddddIHH", wkb.read(60)
    )  # raster Metadata

    for band in range(bands):

        bits = int(struct.unpack(endian + "b", wkb.read(1))[0])  # raster header band attributes
        if bool(bits & 128):
            sys.exit("OffLine PostGIS WKB Data not supported.")

        size = [1, 1, 1, 1, 1, 2, 2, 4, 4, 4, 8][bits & 15]
        dtype = ["b1", "u1", "u1", "i1", "u1", "i2", "u2", "i4", "u4", "f4", "f8"][bits & 15]

        wkb.read(size)  # Skip raster NoData value

        if band == 0:
            out = np.zeros((height, width, bands), dtype=np.dtype(dtype))
            pixtype = bits & 15
        elif pixtype != bits & 15:
            sys.exit("Mixed PostGIS WBK Data type not supported.")

        out[:, :, band] = np.ndarray((height, width), buffer=wkb.read(width * height * size), dtype=np.dtype(dtype))

    return out


def write_tile(root, tile, colors, out):
    """ """
    os.makedirs(os.path.join(root, str(tile.z), str(tile.x)), exist_ok=True)

    out_path = os.path.join(root, str(tile.z), str(tile.x))
    os.makedirs(out_path, exist_ok=True)

    out = Image.fromarray(out, mode="P")
    out.putpalette(complementary_palette(make_palette(colors[0], colors[1])))
    out.save(os.path.join(out_path, "{}.png".format(tile.y)), optimize=True)


def main(args):

    if (args.geojson and args.postgis) or (not args.geojson and not args.postgis):
        sys.exit("Input features to rasterize must be either GeoJSON or PostGIS")

    config = load_config(args.config)
    tile_size = args.tile_size if args.tile_size else config["model"]["tile_size"]
    colors = [classe["color"] for classe in config["classes"]]
    burn_value = 1

    os.makedirs(args.out, exist_ok=True)
    log = Logs(os.path.join(args.out, "log"), out=sys.stderr)

    def geojson_parse_polygon(feature_map, polygon, i):

        try:
            for i, ring in enumerate(polygon["coordinates"]):  # GeoJSON coordinates could be N dimensionals
                polygon["coordinates"][i] = [[x, y] for point in ring for x, y in zip([point[0]], [point[1]])]

            for tile in burntiles.burn([{"type": "feature", "geometry": polygon}], zoom=args.zoom):
                feature_map[mercantile.Tile(*tile)].append({"type": "feature", "geometry": polygon})

        except ValueError:
            log.log("Warning: invalid feature {}, skipping".format(i))

        return feature_map

    def geojson_parse_geometry(feature_map, geometry, i):

        if geometry["type"] == "Polygon":
            feature_map = geojson_parse_polygon(feature_map, geometry, i)

        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                feature_map = geojson_parse_polygon(feature_map, {"type": "Polygon", "coordinates": polygon}, i)
        else:
            log.log("Notice: {} is a non surfacic geometry type, skipping feature {}".format(geometry["type"], i))

        return feature_map

    if args.geojson:

        if not all(tile.z == args.zoom for tile in tiles_from_csv(args.cover)):
            sys.exit("With GeoJson input, zoom level and cover tiles z values have to be the same.")

        feature_map = collections.defaultdict(list)
        epsg = 4326
        # Compute a spatial index like
        for geojson_file in args.geojson:
            with fiona.open(geojson_file) as feature_collection:
                epsg = feature_collection.crs['init'][5:]
                for i, feature in enumerate(tqdm(feature_collection, ascii=True, unit="feature")):

                    if feature["geometry"]["type"] == "GeometryCollection":
                        for geometry in feature["geometry"]["geometries"]:
                            feature_map = geojson_parse_geometry(feature_map, geometry, i)
                    else:
                        feature_map = geojson_parse_geometry(feature_map, feature["geometry"], i)

        # Rasterize tiles
        for tile in tqdm(list(tiles_from_csv(args.cover)), ascii=True, unit="tile"):
            if tile in feature_map:
                out = geojson_tile_burn(tile, feature_map[tile], tile_size, burn_value, epsg)
            else:
                out = np.zeros(shape=(tile_size, tile_size), dtype=np.uint8)

            write_tile(args.out, tile, colors, out)

    if args.postgis:

        try:
            pg_conn = psycopg2.connect(config["dataset"]["pg_dsn"])
            pg = pg_conn.cursor()
        except Exception:
            sys.exit("Unable to connect PostgreSQL: {}".format(config["dataset"]["pg_dsn"]))

        try:
            pg.execute("SELECT ST_Srid(geom) AS srid FROM ({} LIMIT 1) AS sub".format(args.postgis))
            srid = pg.fetchone()[0]
        except Exception:
            sys.exit("Unable to retrieve geometry SRID.")

        for tile in tqdm(list(tiles_from_csv(args.cover)), ascii=True, unit="tile"):

            s, w, e, n = mercantile.bounds(tile)
            raster = np.zeros((tile_size, tile_size))

            query = """
WITH
     bbox      AS (SELECT ST_Transform(ST_MakeEnvelope({},{},{},{}, 4326), {}  ) AS bbox),
     bbox_merc AS (SELECT ST_Transform(ST_MakeEnvelope({},{},{},{}, 4326), 3857) AS bbox),

     rast_a    AS (SELECT ST_AddBand(
                           ST_SetSRID(
                             ST_MakeEmptyRaster({}, {}, ST_Xmin(bbox), ST_Ymax(bbox), (ST_YMax(bbox) - ST_YMin(bbox)) / {}),
                           3857),
                          '8BUI'::text, 0) AS rast
                   FROM bbox_merc),

     features  AS (SELECT ST_Union(ST_Transform(ST_Force2D(geom), 3857)) AS geom
                   FROM ({}) AS sub, bbox
                   WHERE ST_Intersects(geom, bbox)),

     rast_b    AS (SELECT ST_AsRaster(geom, rast, '8BUI', {}) AS rast
                   FROM features, rast_a
                   WHERE NOT ST_IsEmpty(geom))

SELECT ST_AsBinary(ST_MapAlgebra(rast_a.rast, rast_b.rast, '{}', NULL, 'FIRST')) AS wkb FROM rast_a, rast_b

""".format(
                s, w, e, n, srid, s, w, e, n, tile_size, tile_size, tile_size, args.postgis, burn_value, burn_value
            )

            try:
                pg.execute(query)
                row = pg.fetchone()
                if row:
                    raster = np.squeeze(wkb_to_numpy(io.BytesIO(row[0])), axis=2)

            except Exception:
                log.log("Warning: Invalid geometries, skipping {}".format(tile))
                pg_conn = psycopg2.connect(config["dataset"]["pg_dsn"])
                pg = pg_conn.cursor()

            write_tile(args.out, tile, colors, raster)

    if args.web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "./"
        tiles = [tile for tile in tiles_from_csv(args.cover)]
        web_ui(args.out, base_url, tiles, tiles, "png", template)
