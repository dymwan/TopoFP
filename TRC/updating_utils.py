import os
from osgeo import ogr, gdal
from basicTools import timeRecord
from tqdm import tqdm

def get_srs(shp_dir):
    ref_ds = ogr.Open(shp_dir)
    ref_lyr = ref_ds.GetLayer()
    srs = ref_lyr.GetSpatialRef()
    return srs




@timeRecord
def merge_shps_to_one(in_shps:list, dst_shp, ref_shp=None):

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

    # featureDefn = outLayer.GetLayerDefn()
    if ref_shp is not None:
        ref_srs = get_srs(ref_shp)
        
        translate = True
    else:
        translate = False

    for lyr in shp_lyrs:
        for feat_i in lyr:
            outLayer.CreateFeature(feat_i)
    


@timeRecord
def Union(in_shp, dst_shp):
    ds = ogr.Open(in_shp, 1 )
    ref_layer = ds.GetLayer(0)

    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(dst_shp):
            shpDriver.DeleteDataSource(dst_shp)
    outDataSource = shpDriver.CreateDataSource(dst_shp)
    union_Layer = outDataSource.CreateLayer(
        'union', 
        geom_type=ogr.wkbMultiPolygon, 
        srs=ref_layer.GetSpatialRef()
        )

    featureDefn = union_Layer.GetLayerDefn()

    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
    
    for feat in ref_layer:
        multipoly.AddGeometry(feat.geometry())
    
    union_geom = multipoly.UnionCascaded()

    out_feat = ogr.Feature(featureDefn)
    out_feat.SetGeometry(union_geom)
    union_Layer.CreateFeature(out_feat)
    union_Layer.SyncToDisk()


def create_shp(dst_dir, srs, vtype=ogr.wkbMultiPolygon):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(dst_dir)
    dst_lyr = dst_ds.CreateLayer('dst_lyr', srs, vtype)
    dst_lyr.SyncToDisk()
    return dst_ds
    

@timeRecord
def is_to_not(target_shp, current_shp):
    target_ds = ogr.Open(target_shp)
    target_lyr = target_ds.GetLayer()
    target_srs = target_lyr.GetSpatialRef()

    cur_ds = ogr.Open(current_shp, 1)
    cur_lyr = cur_ds.GetLayer()
    cur_srs = cur_lyr.GetSpatialRef()
    pred_feat = cur_lyr.GetNextFeature()
    pred_geom = pred_feat.geometry()
    
    if not target_srs.IsSame(cur_srs):
        pred_geom.TransformTo(target_srs)
        feat = ogr.Feature(target_lyr.GetLayerDefn())
        feat.SetGeometry(pred_geom)
        cur_lyr.CreateFeature(feat)
        
        

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(current_shp.replace('.shp', '.lyrIn.shp'))
    dst_lyr = dst_ds.CreateLayer('dst_lyr', target_srs, ogr.wkbMultiPolygon)

    intersection_lyr = cur_lyr.Intersection(target_lyr, dst_lyr)
    dst_lyr.SyncToDisk()
    print(dst_lyr.GetFeatureCount())

def _show_extent(shp_dir):
    ds = ogr.Open(shp_dir)
    lyr = ds.GetLayer()
    print(lyr.GetExtent())

def intersect_onebyone(src_shp, pred_shp):

    src_ds = ogr.Open(src_shp)
    src_lyr = src_ds.GetLayer()
   

    pred_ds = ogr.Open(pred_shp)
    pred_lyr = pred_ds.GetLayer()
    
    a = 0
    for feat in src_lyr:
        src_geom = feat.geometry()
        pred_lyr.SetSpatialFilter(src_geom)
        print(src_geom)
        print(pred_lyr.GetExtent())
        print(pred_lyr.GetFeatureCount())
        a+=1
        if a>20:
            exit()
        # for pred_feat in pred_lyr:
        #     print(pred_feat.GetFeatureCount())
            # if src_geom.Intersect(pred_feat.geometry()):
            #     a += 1
            # else:
            #     a += 1



if __name__ == '__main__':
    
    target_shp = r'D:\Updation\target\ELCLTB371521.shp'
    test_src = r'D:\Updation\pred'
    merged_shp = r'D:\Updation\updating\merged.shp'
    merged_shpt = r'D:\Updation\updating\merged.translate.shp'
    pred_union = r'D:\Updation\updating\pred_union.shp'
    
    # _show_extent(merged_shp)
    
    # exit()

    # merge multiple layer
    in_shps = [os.path.join(test_src, name) for name in os.listdir(test_src) if name.endswith('.shp')]
    # out_shps = os.path.join(test_src, 'merged.shp')
    merge_shps_to_one(in_shps, merged_shp, ref_shp=target_shp)
    
    _show_extent(target_shp)
    _show_extent(merged_shp)


    options = gdal.VectorTranslateOptions(
		format=format,
		accessMode=accessMode,
		srcSRS=spatial,
		dstSRS=dstSRS,
		reproject=True,
		selectFields=selectFields,
		layerName=layerName,
		geometryType=geometryType,
		dim=dim
	)
	gdal.VectorTranslate(
		destDataPath,
		srcDS=shapeFilePath,
		options=options
	)

    
    # intersect_onebyone(target_shp, merged_shp)


    # Union(merged_shp, pred_union)
    
    # is_to_not(target_shp, pred_union)

    