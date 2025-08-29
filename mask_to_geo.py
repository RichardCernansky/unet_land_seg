import os
import cv2
import numpy as np
from osgeo import gdal, ogr, osr

def png_gray_to_geotiff(png_path, ref_tif_path, dst_tif_path):
    """Convert a grayscale PNG mask (0=background, >0=foreground) into a georeferenced GeoTIFF."""
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read {png_path}")

    # Convert to binary: any non-zero → 1 - postprocessed ALWAYS with only one class
    class_mask = (img > 0).astype(np.uint8)

    # Use georeference from the reference GeoTIFF
    ref = gdal.Open(ref_tif_path, gdal.GA_ReadOnly)
    gt = ref.GetGeoTransform()
    proj = ref.GetProjection()
    h, w = class_mask.shape

    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(
        dst_tif_path, w, h, 1, gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"]
    )
    out.SetGeoTransform(gt)
    out.SetProjection(proj)

    band = out.GetRasterBand(1)
    band.WriteArray(class_mask)
    band.SetNoDataValue(0)
    band.FlushCache()

    out = None
    ref = None
    print(f"✅ Saved georeferenced GeoTIFF: {dst_tif_path}")

def geotiff_to_shapefile(geotiff_path, shp_path):
    """Polygonize non-zero areas of a GeoTIFF into a Shapefile."""
    src = gdal.Open(geotiff_path, gdal.GA_ReadOnly)
    band = src.GetRasterBand(1)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        drv.DeleteDataSource(shp_path)
    ds = drv.CreateDataSource(shp_path)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(src.GetProjection())

    layer = ds.CreateLayer("mask", srs=srs, geom_type=ogr.wkbPolygon)
    fld = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(fld)

    # Polygonize whole raster → creates polygons for 0 and 1
    gdal.Polygonize(band, None, layer, 0, [], callback=None)

    # Delete polygons where value == 0 (background)
    to_delete = [f.GetFID() for f in layer if f.GetField("value") == 0]
    for fid in to_delete:
        layer.DeleteFeature(fid)

    layer.SyncToDisk()
    layer = None
    ds = None
    src = None
    print(f"✅ Saved Shapefile: {shp_path}")

if __name__ == "__main__":
    png_in = "./data/predicted_masks/Buriny_post.png"   # grayscale mask
    ref_tif = "/home/ramexvpn/unet_land_seg/data/images/Buriny.tif"
    tif_out = "./data/predicted_masks/Buriny_post.tif"
    shp_out = "./data/predicted_masks/Buriny_post.shp"

    png_gray_to_geotiff(png_in, ref_tif, tif_out)
    geotiff_to_shapefile(tif_out, shp_out)
