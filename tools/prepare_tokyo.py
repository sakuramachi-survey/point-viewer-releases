"""Crop the two Tokyo LP demonstration datasets using the publisher's tile index."""
import gzip
import hashlib
import json
import math
import re
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import laspy
import mapbox_vector_tile
import mercantile
import numpy as np
import requests
from pyproj import CRS, Transformer
from scipy import ndimage
from shapely.geometry import MultiPoint, box, shape

SOURCE = "https://gic-tokyo.s3.ap-northeast-1.amazonaws.com"
INDEX = SOURCE + "/2024/dig/Vectortile/23ku/lp"
OFFICIAL = "https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024"
SITES = [
    ("tokyo-metropolitan-gov", 139.69172, 35.68963, 80),
    ("tokyo-dome", 139.75192, 35.70564, 125),
]
to_en = Transformer.from_crs(4326, 6677, always_xy=True)
to_ll = Transformer.from_crs(6677, 4326, always_xy=True)
to_web = Transformer.from_crs(6677, 3857, always_xy=True)
session = requests.Session()


def tile_links(bounds):
    west, south, east, north = to_ll.transform_bounds(*bounds)
    clip = box(*to_web.transform_bounds(*bounds))
    for zoom in (15, 14, 16, 13, 12, 17, 11, 10):
        urls = set()
        for tile in mercantile.tiles(west, south, east, north, zoom):
            addr = f"{INDEX}/{tile.z}/{tile.x}/{tile.y}.pbf"
            response = session.get(addr, timeout=45)
            print("index", response.status_code, addr, flush=True)
            if response.status_code in (403, 404):
                continue
            response.raise_for_status()
            raw = response.content
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            tile_box = mercantile.xy_bounds(tile)
            for layer in mapbox_vector_tile.decode(raw).values():
                extent = layer["extent"]
                def scale_xy(x, y):
                    return ((x-tile_box.left)*extent/(tile_box.right-tile_box.left),
                            (y-tile_box.bottom)*extent/(tile_box.top-tile_box.bottom))
                left, bottom, right, top = clip.bounds
                tile_clip = box(*scale_xy(left, bottom), *scale_xy(right, top))
                for feature in layer["features"]:
                    if not shape(feature["geometry"]).intersects(tile_clip):
                        continue
                    props = str(feature["properties"])
                    refs = re.findall(r'https?://[^\s"<>{}\x27]+?\.zip', props, re.I)
                    if not refs:
                        raise ValueError("No download link in matching tile properties: " + props[:1100])
                    for ref in refs:
                        if urlsplit(ref).hostname not in (
                            "gic-tokyo.s3.ap-northeast-1.amazonaws.com", "gic-tokyo.s3.amazonaws.com"
                        ):
                            raise ValueError("Unexpected publisher URL: " + ref)
                        urls.add(ref.replace("http://", "https://", 1))
        if urls:
            return sorted(urls)
    raise ValueError("No intersecting Tokyo LP tiles")


def fetch_crop(urls, bounds, temp):
    blocks = []
    for i, url in enumerate(urls):
        print("source", url, flush=True)
        path = temp / f"{i}.zip"
        with session.get(url, stream=True, timeout=160) as resp, path.open("wb") as dest:
            resp.raise_for_status()
            for chunk in resp.iter_content(1024*1024):
                dest.write(chunk)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith((".las", ".laz")):
                    continue
                with archive.open(name) as source, laspy.open(source) as las:
                    crs = las.header.parse_crs()
                    if crs and crs.to_epsg() not in (6677, None):
                        raise ValueError(f"Unexpected source CRS: {crs}")
                    for chunk in las.chunk_iterator(250000):
                        x, y = np.asarray(chunk.x), np.asarray(chunk.y)
                        keep = ((x>=bounds[0]) & (x<=bounds[2]) & (y>=bounds[1]) & (y<=bounds[3]))
                        if not np.any(keep):
                            continue
                        xyz = np.stack((x[keep], y[keep], np.asarray(chunk.z)[keep]), axis=1)
                        colors = (np.stack((chunk.red[keep],chunk.green[keep],chunk.blue[keep]), axis=1)
                                  if "red" in chunk.point_format.dimension_names else np.zeros_like(xyz))
                        blocks.append(np.column_stack((xyz,chunk.classification[keep],colors)))
        path.unlink()
    if not blocks:
        raise ValueError("No survey points in the projected crop")
    return np.concatenate(blocks)


def make_spot(key, lon, lat, radius, out):
    cx, cy = (round(v) for v in to_en.transform(lon, lat))
    bounds = [cx-radius, cy-radius, cx+radius, cy+radius]
    urls = tile_links(bounds)
    with tempfile.TemporaryDirectory() as tmp:
        pts = fetch_crop(urls, bounds, Path(tmp))
    pts = pts[np.isfinite(pts[:,2]) & (pts[:,2] > -5) & (pts[:,2] < 360)]
    x,y,z,cls = pts[:,:4].T
    # Verify every interior quadrant has survey returns; incomplete tiles cannot be released.
    for dx in (-1,1):
        for dy in (-1,1):
            if np.sum(((x-cx)*dx > 0) & ((y-cy)*dy > 0)) < 2000:
                raise ValueError("Incomplete crop")
    ground_points = z[(cls == 2) & (np.maximum(np.abs(x-cx),np.abs(y-cy)) >= radius*.72)]
    if len(ground_points) < 50:
        raise ValueError("Insufficient classified perimeter ground")
    ground = round(float(np.median(ground_points)), 2)
    # Retain the connected raised footprint nearest the centre, excluding adjacent buildings.
    high = z > ground + (25 if key == "tokyo-dome" else 50)
    grid = np.zeros((2*radius+1,2*radius+1),dtype=bool)
    xi = np.floor(x-bounds[0]).astype(int)
    yi = np.floor(y-bounds[1]).astype(int)
    grid[yi[high], xi[high]] = True
    components, count = ndimage.label(ndimage.binary_closing(grid,iterations=2))
    scores = np.bincount(components[radius-20:radius+21,radius-20:radius+21].ravel(),minlength=count+1)
    if not scores[1:].any():
        raise ValueError("Landmark not found")
    footprint = components == (1+np.argmax(scores[1:]))
    iy,ix = np.where(footprint)
    hull = MultiPoint(np.stack((ix+bounds[0]+.5,iy+bounds[1]+.5),axis=1)).convex_hull
    # A modest margin includes roof eaves in the approximate exercise region.
    poly = hull.buffer(3,join_style=2).simplify(1.5).intersection(box(*bounds))
    ring = [[round(a,2),round(b,2)] for a,b in list(poly.exterior.coords)[:-1]]
    within = footprint[yi,xi]
    meta = dict(roiBounds=dict(zip(("minX","minY","maxX","maxY"),bounds)), region=ring,
                referenceElevation=ground, center=[cx,cy], sourceTiles=urls)
    if key.endswith("gov"):
        bands = []
        for lower, upper in [(ground+35, ground+175),(ground+175,ground+252)]:
            roof = z[within & (z>=lower) & (z<=upper)]
            if len(roof)<100:
                raise ValueError("Missing roof band")
            hist, bins = np.histogram(roof,bins=np.arange(math.floor(lower),math.ceil(upper)+1))
            bands.append(float(bins[np.argmax(hist)]+.5))
        meta["heights"] = bands
    # One original return per 0.35 m XY / 0.5 m Z voxel. Coordinates stay in survey metres.
    cells=np.floor((pts[:,:3]-[bounds[0],bounds[1],0])/[.35,.35,.5]).astype(np.int32)
    keep=np.sort(np.unique(cells,axis=0,return_index=True)[1])
    pts=pts[keep]
    header=laspy.LasHeader(point_format=2,version="1.2")
    header.scales=np.array([.001,.001,.001])
    header.offsets=np.array([cx,cy,0])
    header.creation_date=date(2026,11,20)
    header.add_crs(CRS.from_epsg(6677))
    las=laspy.LasData(header)
    las.x,las.y,las.z=pts[:,:3].T
    las.classification=pts[:,3].astype(np.uint8)
    las.red,las.green,las.blue=pts[:,4:7].astype(np.uint16).T
    las.return_number=np.ones(len(pts),dtype=np.uint8)
    las.number_of_returns=np.ones(len(pts),dtype=np.uint8)
    filename=out/(key+"-v1.laz")
    las.write(filename)
    raw=filename.read_bytes()
    if not (20000 <= len(pts) and len(raw) <= 50*1024**2):
        raise ValueError("Point count or file size outside sample limits")
    with laspy.open(filename) as check:
        assert check.header.parse_crs().to_epsg()==6677 and check.header.point_count==len(pts)
        meta.update(minXYZ=list(check.header.mins),maxXYZ=list(check.header.maxs))
    meta.update(id=key,filename=filename.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),points=len(pts))
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    return meta


if __name__ == "__main__":
    out=Path("samples/v1")
    out.mkdir(parents=True,exist_ok=True)
    metadata=[make_spot(*item,out) for item in SITES]
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
