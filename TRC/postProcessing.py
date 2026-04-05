from operator import mod
from sre_constants import LITERAL_LOC_IGNORE
import numpy as np
import torch

from .algorithm import *
from .basicTools import *
from .utils import *
# from .Features import Line, Node
import argparse, json
from osgeo import gdal, ogr, osr

from .prepocessing import post_process



def get_anchor(x,y, stride):
    nx = x// stride
    ny = y// stride
    
    nx=  nx +1 if x % stride > 0 else nx
    ny=  ny +1 if y % stride > 0 else ny

    anchors = []
    for xi in range(nx):
        for yi in range(ny):
            x_start = xi * stride
            y_start = yi * stride
            x_offset = stride if x_start + stride <=  x else x -x_start-1
            y_offset = stride if y_start + stride <=  y else y -y_start-1

            anchors.append([x_start,
                y_start,
                x_offset,
                y_offset,
                xi,
                yi
                ])
    return anchors

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

def update_geotransform(gt, xs, ys):
    #[12898708.0929, 
    # 1.9999291190106971, 
    # 0.0, 
    # 4335239.1283, 
    # 0.0, 
    # -2.0000829286914583]

    print('before', gt)
    gt[0] += xs * gt[1]
    gt[3] += ys * gt[5]
    print('after', gt)
    return gt


def getArgs():
    parser = argparse.ArgumentParser(prog='UAV_inference', prefix_chars= '-', \
        description='This script is designed for pytorch model inference.',\
        epilog= 'Contact dymwan@gmail.com')
    
    # parser.add_argument("-h", "--help", action="help", help="")

    parser.add_argument('-c', '--config', metavar='CONFIG', type=str, default=r'./test.json', help='Inference parameters configuration.', dest='cfg')
    
    return parser.parse_args()

def parse_json(json_file):
    with open(json_file, 'r') as jf:
        return json.load(jf) 


def get_device(device_number):
    if device_number >= 0:
        device = torch.device('cuda:%s' % int(device_number)\
             if torch.cuda.is_available() else 'cpu')
    elif device_number == -1:
        device = torch.device('cpu')
    else:
        raise ValueError('define the device with number for GPU or -1 for CPU')
    
    return device

def closure_field(imarr):
    field_bin = getIsolateCls(imarr, 3)
    field_bin_dilate = cv2.dilate(field_bin, kernel=np.ones([3,3], np.uint8), iterations=2)
    field_gap = field_bin_dilate - field_bin
    imarr[(imarr == 4) & (field_gap == 1)] = 2
    return imarr

def post_process_single_folder(pred_dir, dst_folder, work_env, device, logger, stride=5000):

    
    print(pred_dir)
    basename = os.path.basename(pred_dir).split('.')[0]
    
    dst_shp=os.path.join(dst_folder, basename+'.shp')
    if os.path.isfile(dst_shp):
        print(f'file {dst_shp} exists, continue... ')
        return dst_shp

    # STRIDE

    ds = gdal.Open(pred_dir)
    x,y = ds.RasterXSize, ds.RasterYSize
    
    if x > 10000 or y > 10000:
        print('due to the memory limit, stride is set to 5000')
        stride = 5000


    anchors = get_anchor(x, y, stride)
    
    mid_shps = []
    
    for xs, ys, xoff, yoff, xi, yi in anchors:
        # continue

        tmp_arr = ds.ReadAsArray(xs, ys, xoff, yoff)
        tmp_arr = closure_field(tmp_arr)
        
        arr_uniq = np.unique(tmp_arr[...])
        if 2 not in arr_uniq or 3 not in arr_uniq:
            continue


        gtxy = list(ds.GetGeoTransform())

        gtxy = update_geotransform(gtxy, xs, ys)

    
        driver_type ='MEM'
        mid_dst_dir = ""

        tmp_driver = gdal.GetDriverByName(driver_type)
        # tmp_driver = gdal.GetDriverByName('GTiff')
        tmp_raster = tmp_driver.Create(
            mid_dst_dir, xoff+2, yoff+2, 1, gdal.GDT_Int32)
        # tmp_raster = tmp_driver.Create(
        #     r'E:\pytorch_dym\test_data\predicted\final\temp_{}_{}.tif'.format(xi, yi), xoff, yoff, 1, gdal.GDT_Int32)

        tmp_raster.SetGeoTransform(gtxy)
        tmp_raster.SetProjection(ds.GetProjection())

        
        
        filled = post_process(tmp_arr, device)
        filled = eliminate_bound(filled, )
        # filled = np.pad(filled, ((1,0),(1,0)), mode='edge')
        tmp_raster.GetRasterBand(1).WriteArray(filled.astype(np.int32))
        
        name = f'{xi}_{yi}'
        dst_shp_xy = os.path.join(work_env, f'{basename}_{xi}_{yi}.shp')
        
        raster2poly(tmp_raster, dst_shp_xy, name)
        # tmp_raster.Close()
        tmp_raster.FlushCache()
        tmp_driver = None
        tmp_raster = None

        mid_shps.append(dst_shp_xy)

    
    merge_shps_to_one(mid_shps, dst_shp)
    logger.info(f'finished processing file {pred_dir}')
    return dst_shp
    


def run_post_pro(pred_dir, work_root, dst_folder, device, logger):

    if isinstance(pred_dir, str):
        if os.path.isdir(pred_dir):
            for filename in os.listdir(pred_dir):
                if not filename.endswith(cfg.get("suffix")):
                    continue
                else:
                    pred_file_dir = os.path.join(pred_dir, filename)
                    return post_process_single_folder(pred_file_dir, dst_folder, work_root, device, logger)
        elif os.path.isfile(pred_dir):
            logger.info('start doing post-processing on %s' % pred_dir)
            return post_process_single_folder(pred_dir, dst_folder, work_root, device, logger)
    
    elif isinstance(pred_dir, list):
        produced_shp = []
        for sub_pred_dir in pred_dir:
            produced_shp.append(run_post_pro(sub_pred_dir, work_root, dst_folder, device, logger))
        return produced_shp
        

if __name__ == '__main__':

    args = getArgs()
    if not os.path.isfile(args.cfg):
        raise Exception('Can not load inference configuration: [%s]' % args.cfg)
    
    cfg = parse_json(args.cfg)


    pred_dir = cfg.get("input_dir")
    work_root = cfg.get("work_env")
    dst_folder = cfg.get("dst_folder")
    
    run_post_pro(pred_dir, work_root, dst_folder)
    

    
    
