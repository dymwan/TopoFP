
import numpy as np
from osgeo import ogr

from algorithm import *
from basicTools import *
from utils import *
from Features import Line, Node
from tqdm import tqdm

from prepocessing import post_process, update_geotransform, get_anchor

src_folders = [r'/home4/dymwan/heilongjiang_pred_4']
dst_folders = [r'/home4/dymwan/heilongjiang_production']

tmp_dir = '/home4/dymwan/tmp'
STRIDE = 5000


@timeRecord
def merge_shps_to_one(in_shps:list, dst_shp):

    shp_dss = [ogr.Open(in_shp) for in_shp in in_shps]
    shp_lyrs= [ds.GetLayer() for ds in shp_dss]

    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(dst_shp):
            shpDriver.DeleteDataSource(dst_shp)
    outDataSource = shpDriver.CreateDataSource(dst_shp)
    outLayer = outDataSource.CreateLayer(
        dst_shp, 
        geom_type=ogr.wkbMultiPolygon, 
        srs=shp_lyrs[0].GetSpatialRef()
        )


    for lyr in shp_lyrs:
        for feat_i in lyr:
            outLayer.CreateFeature(feat_i)

for src_folder, dst_folder in zip(src_folders, dst_folders):
    
    for fn in os.listdir(src_folder):
        basename = fn.rstrip('.tif')
        if not fn.endswith('.tif'): continue

        fn_tmp_dir = build_dir(tmp_dir, basename)
        
        img_dir = build_dir(src_folder, fn)
        dst_dir = build_dir(dst_folder, fn.replace('.tif', '.shp'))
        
        ds = gdal.Open(img_dir)
        # raw_gt = list(ds.GetGeoTransform())

        X = ds.RasterXSize
        Y = ds.RasterYSize

        anchors = get_anchor(X, Y, STRIDE)
        
        tmp_shps = []
        for xs, ys, xoff, yoff, xi, yi in anchors:
            gtxy = list(ds.GetGeoTransform())
            gtxy = update_geotransform(gtxy, xs, ys)

            tmp_driver = gdal.GetDriverByName('MEM')
            tmp_raster = tmp_driver.Create("", xoff, yoff, 1, gdal.GDT_Int32)
            tmp_raster.SetGeoTransform(gtxy)
            tmp_raster.SetProjection(ds.GetProjection())



            tmp_arr = ds.ReadAsArray(xs, ys, xoff, yoff)

            filled = post_process(tmp_arr)
            filled = eliminate_bound(filled, )
            tmp_raster.GetRasterBand(1).WriteArray(filled.astype(np.int32))
            name = f'{xi}_{yi}'
            dst_shp_xy = os.path.join(fn_tmp_dir, f'{basename}_{xi}_{yi}.shp')
            
            raster2poly(tmp_raster, dst_shp_xy, name)
            tmp_shps.append(dst_shp_xy)

            tmp_raster.FlushCache()
            tmp_driver = None
            tmp_raster = None
        
        merge_shps_to_one(tmp_shps, dst_dir)


        
