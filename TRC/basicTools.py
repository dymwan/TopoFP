import os
import shutil
# from turtle import numinput
import numpy as np
import cv2

from osgeo import gdal



SCALE_FLAG = {
    'INTER_NEAREST' : cv2.INTER_NEAREST,
    'INTER_LINEAR' : cv2.INTER_LINEAR,
    'INTER_AREA' : cv2.INTER_AREA,
    'INTER_CUBIC' : cv2.INTER_CUBIC,
    'INTER_LANCZOS4' : cv2.INTER_LANCZOS4,
}


def timeRecord(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{:>25} >> costs Time (sec): {:.2f}'.format(func.__name__, (end_time - start_time)))
        return result
    return wrapper

def getIsolateCls(im, *cls_codes):
    
    out_im = np.zeros(im.shape, dtype=np.uint8)
    for cls_code in cls_codes:
        out_im[im == cls_code] = 1

    return out_im

def getIndivCls(im, *cls_codes):
    _bin =  getIsolateCls(im, *cls_codes)
    
    num_indiv, indiv = cv2.connectedComponents(_bin)
    return num_indiv, indiv

def getIndivClsWithStats(im, *cls_codes, **kwargs):
    connectivity = kwargs.get('connectivity')
    connectivity = 8 if connectivity is None else connectivity

    _bin = getIsolateCls(im, *cls_codes)
    n_indiv, indiv_non_bound, stats, centroids = cv2.connectedComponentsWithStats(_bin, connectivity=connectivity)
    return n_indiv, indiv_non_bound, stats, centroids


def subset_info_filter(ds, subset_info):
    if subset_info[0] > ds.RasterXSize or subset_info[1] > ds.RasterYSize:
        raise Exception('x, y start in subset_info must be inside the raster extent\
            you enter the x,y starts are [ %s, %s ], and the size of the raster is \
                [ %s, %s ]' % (subset_info[0], subset_info[1], ds.RasterXSize, ds.RasterYSize))

    if subset_info[0] + subset_info[2] > ds.RasterXSize:
        subset_info[2] = ds.RasterXSize - subset_info[0]
        
    if subset_info[1] + subset_info[3] > ds.RasterYSize:
        subset_info[3] = ds.RasterYSize - subset_info[1]
    
    return subset_info

def updateGeoTransform(ds, subset_info=None, scale=None):
    '''update GeoTransform of gdal.Datasource'''
    gt = list(ds.GetGeoTransform())

    if scale is not None:
        gt[1] /= scale
        gt[5] /= scale
    else:
        scale = 1

    
    if subset_info is not None:
        gt[0] += subset_info[0] * gt[1] * scale
        gt[3] += subset_info[1]* gt[5] * scale
    
    ds.SetGeoTransform(gt)
    return ds



def loadGeoRaster(src_dir:str, scale:float=None, return_ds=False,
            subset_info:list=None, single_band:int=None, scale_method:str=None):
    '''load entire or part of raster data as array
    
    Args:
        -src_dir: the directory of raster file, support file format:
            -- TIFF
            -- IMG
            TODO the file format has sub-datasource (hd5)
        
        -scale [optional]: the factor to zoom raster data, now support fixed scale,
        which means the size of loaded is divided scale by original size
        
        -subset_info [optional]: a four-elements list like 
                        [start_x, start_y, offset_x, offset_y]
        
        -single_band [optional]: load only bands from multi-bands raster

        -scale_method [optional]: interplote method, including 'INTER_NEAREST',\
                'INTER_LINEAR','INTER_AREA','INTER_CUBIC','INTER_LANCZOS4', \
                default is INTER_NEAREST

    Returns:
        loaded_arr: A ndarray with shape as [band, height, width]
        
        gdal.datasource: a fixed gdal.datasource according to scale and 
                            subset_info

    Notes:
        1. It is stipilated here that
            gdal.datasource.RasterXsize -> width -> ndarray.shape[-1] -> gt[0] -> subset_info[0]
            gdal.datasource.RasterYsize -> height-> ndarray.shape[-2] -> gt[3] -> subset_info[0]
        2. In the laoding process, subset first and then scale
        3. single_band start from 1, end to the band number of the raster
    '''

    if not os.path.isfile(src_dir):
        raise FileExistsError(': [%s]' % src_dir)
    
    ds = gdal.OpenShared(src_dir, 0) # mode 0 -> readonly, 1 -> writeble

    # if ds is None:
    #     

    # print(ds.GetGeoTransform())
    # TODO if subset_info overflow
    subset_info = [0, 0, ds.RasterXSize, ds.RasterYSize] if \
        subset_info is None else subset_info_filter(ds, subset_info)

    # print(subset_info)

    if isinstance(single_band, int) and single_band>0 and single_band<=ds.RasterCount:
        raster_array = ds.GetRasterBand(single_band).ReadAsArray(*subset_info)
    else:
        raster_array = ds.ReadAsArray(*subset_info)
    
    # 
    #     raster_array = raster_array.reshape(1, *raster_array.shape)
    #     print(raster_array.shape)

    if isinstance(scale, float):
        scale_method = SCALE_FLAG[scale_method] if scale_method is not None else cv2.INTER_LINEAR
        if len(raster_array.shape) > 2:
            raster_array = raster_array.transpose(1,2,0)
        raster_array = cv2.resize(
                raster_array, 
                (0,0), 
                fx=scale, 
                fy=scale, 
                interpolation=scale_method
            )
        if len(raster_array.shape) > 2:
            raster_array = raster_array.transpose(2,0,1)

    # if subset_info or scale:
    #     ds = updateGeoTransform(ds, subset_info=subset_info, scale=scale)
    # print(ds.GetGeoTransform())
    if return_ds:
        return raster_array, ds
    else:
        return raster_array



def WriteTiff(img_arr, dst_dir, ds=None, gt=None, dst_fileformat='tif', 
    size_source='in_arr', dtype = None):
    
    io_dtype = {
        gdal.GDT_Byte: np.dtype(np.uint8),
        gdal.GDT_UInt16: np.dtype(np.uint16),
        gdal.GDT_UInt32: np.dtype(np.uint32),
        gdal.GDT_Float32: np.dtype(np.float32),
        gdal.GDT_Float64: np.dtype(np.float64),
        gdal.GDT_Int32: np.dtype(np.int32),
        
    }
    
    oi_dtype = {v:k for k, v in io_dtype.items()}
    oi_dtype[np.dtype(np.byte)] = gdal.GDT_Byte

    dtype = dtype if dtype is not None else \
        oi_dtype[ np.dtype(img_arr.dtype) ]
    
    driver = gdal.GetDriverByName('GTiff')
    if len(img_arr.shape) == 3:
        nb = img_arr.shape[0]
    elif len(img_arr.shape) == 2:
        nb = 1
    else:
        raise ValueError('The size of input tensor has abnormal size in %s.'
            % img_arr.shape)
    
    if size_source == 'in_arr':
        outRaster = driver.Create(
            dst_dir, img_arr.shape[-1], img_arr.shape[-2], nb, dtype)
    elif size_source == 'ref_ds':
        outRaster = driver.Create(
            dst_dir, img_arr.shape[-1], img_arr.shape[-2], nb, dtype)

    if ds is not None:
        outRaster.SetGeoTransform(ds.GetGeoTransform())
        outRaster.SetProjection(ds.GetProjection())
        
    if gt is not None:
        outRaster.SetGeoTransform(gt)

    if dtype is not None and img_arr.dtype != io_dtype[dtype]:
        img_arr = img_arr.astype(io_dtype[dtype])
    
    
    if nb == 1:
        outRaster.GetRasterBand(1).WriteArray(img_arr)
    elif nb > 1:
        for band_idx in range(nb):
            outRaster.GetRasterBand(band_idx + 1).WriteArray(img_arr[band_idx,:,:])
    outRaster.FlushCache()
    Information()(1, f'Tiff file {dst_dir} has been written !')


class InternelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Information:
    # def __init__(self) -> None:
        
    def __call__(self, level, *args, **kwds) -> None:
        assert level in [1,2,3]
        _type_str = {
            1: 'INFO',
            2: 'WARNING',
            3: 'ERROR',
        }.get(level)
        
        info_body = '>>[{}]: '.format(_type_str) + '\n'.join([
            '{}'.format(ag) for ag in args
        ])
        
        if level <3:
            print(info_body)
        else:
            raise InternelError(info_body)


def build_dir(*args, overwrite=False, isfile=False):
    Info = Information()
    _path= os.path.join(*args)
    
    if os.path.isdir(_path) and overwrite:
        shutil.rmtree(_path)
        Info(1, f'Folder {_path} successfully purged!')
    
    if not isfile:
        try:
            os.mkdir(_path)
            Info(1, f'Folder {_path} successfully created!')
        except FileExistsError:
            Info(1, f'Folder {_path} exists, skipping...')
    
    return _path
