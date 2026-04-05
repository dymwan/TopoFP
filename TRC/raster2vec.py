
from tqdm import tqdm
from osgeo import gdalconst, gdal, ogr, osr
from typing import Callable, Iterable, Any, Mapping, List, Optional
import threading
from basicTools import *
from algorithm import *
from utils import *
import os, sys
sys.setrecursionlimit(3000)
import torch

@timeRecord
def raster2poly(raster, outshp):
    inraster = gdal.Open(raster)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备
 
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(raster[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value
    Poly_layer.CreateField(newField)
 
    gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    Polygon.SyncToDisk()
    Polygon = None


def split_list(inlist, nparts):
    list_len = len(inlist)
    gap = list_len // nparts

    sub_lists = []
    
    for pi in range(nparts):
        start = pi* gap
        end = start + gap
        end = end if end < list_len else list_len -1

        sub_lists.append(inlist[start:end])
    return sub_lists

def split_dict(indict, nparts):
    dict_len = len(indict)

    keys = list(indict.keys())
    
    gap = dict_len // nparts
    
    sub_dicts = []
    for pi in range(nparts):
        start = pi* gap
        end = start + gap
        end = end if end < dict_len else dict_len -1
        

        sub_dict = {i: indict[i] for i in keys[start:end]}
        sub_dicts.append(sub_dict)
    
    return sub_dicts
    


class ThreadingPool:
    def __init__(self, func, material, n_threads, args=[], kwargs={}) -> None:
        assert isinstance(material, Iterable)
        
        self.func = func
        self.n_thread = n_threads
        self.args = args
        self.kwargs = kwargs

        self.thread_pool = []
        
        self.material = material
        self.n_material = len(material)
        self.sub_materials = []

        self.initiate_thread_pool()

    def initiate_thread_pool(self):
        if isinstance(self.material, dict):
            self.sub_materials = split_dict(self.material, self.n_thread)
        elif isinstance(self.material, list):
            self.sub_materials = split_list(self.material, self.n_thread)
        else:
            raise TypeError('The material should be dict or list')
        
        for ti in range(self.n_thread):
            # generate_line_vec(line_objs, polylines, indiv_lines_t, indiv_points_t)
            threadi = threading.Thread(
                target  = self.func,
                args    = (self.sub_materials[ti], *self.args),
                kwargs  = self.kwargs #TODO this part may be mistake
            )
            
            self.thread_pool.append(threadi)
    
    def run(self):
        for ti in range(self.n_thread):
            self.thread_pool[ti].start()
        for ti in range(self.n_thread):
            self.thread_pool[ti].join()



def generate_line_vec(line_objs, polylines, indiv_lines_t, indiv_points_t):
    # global indiv_lines_t
    # global indiv_points_t
    # global polylines
    # polylines = []
    for line_code, lineo in tqdm(line_objs.items(), total=len(line_objs), ncols=100):
        # point_codes = lineo.linked_nodes
        point_coords = []
        for pi in lineo.linked_nodes[:2]:
            
            point_coordinate = torch.where(indiv_points_t == pi)
            point_coordinate = torch.cat([*point_coordinate]).view(-1,2)[0]
            point_coords.append(point_coordinate.unsqueeze(0).float())
        
        
        # point_coords = point_coords[:2]

        line_coords = torch.stack([*torch.where(indiv_lines_t == line_code)]).permute(1,0).float()

        

        line_coords = build_sequential_line(point_coords[1], line_coords.float())
        line_coords = DouglasPeuker(line_coords, 2)
        # print(line_coords)
        line_coords = torch.cat([line_coords, point_coords[0]])
        # print(line_coords)
        # exit()
        polylines.append(line_coords.long().cpu().numpy())
        
        print(line_coords, point_coords)
    
    # return polylines

def point_to_line(polylines, shp_dir, ds):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(shp_dir) ## shp文件名称
    srs = osr.SpatialReference(ds.GetProjection())

    log = shp_dir.replace('.shp', '.txt')
    logf = open(log, 'w')

    layer = data_source.CreateLayer("Line", srs, ogr.wkbLineString) ## 图层名称要与shp名称一致
    field_name = ogr.FieldDefn("Name", ogr.OFTString) ## 设置属性
    field_name.SetWidth(20)  ## 设置长度
    layer.CreateField(field_name)  ## 创建字段
    field_length = ogr.FieldDefn("Length", ogr.OFTReal)  ## 设置属性
    layer.CreateField(field_length)  ## 创建字段
    field_n_node = ogr.FieldDefn("n_node", ogr.OFTInteger)  ## 设置属性
    layer.CreateField(field_n_node)  ## 创建字段
    
    x_start, x_unit, _, y_start, __, y_unit = ds.GetGeoTransform()

    for polyline in polylines:
        # print(polyline)
        # print(polyline.shape)
        polyline = polyline.astype(np.float32)
        polyline[:,0] *= x_unit
        polyline[:,0] += x_start
        polyline[:,1] *= y_unit
        polyline[:,1] += y_start
        
        # print(polyline.shape)
        # exit()
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("Name", "line")  ## 设置字段值
        feature.SetField("n_node", len(polyline))  ## 设置字段值
        feature.SetField("Length", "100")  ## 设置字段值
        wkt = 'LINESTRING(%s)' % ', '.join([' '.join([str(e) for e in pair[::-1]]) for pair in polyline])
        logf.write(wkt + '\n')
        line = ogr.CreateGeometryFromWkt(wkt) ## 生成线
        feature.SetGeometry(line)  ## 设置线
        layer.CreateFeature(feature)  ## 添加线

    feature = None ## 关闭属性
    data_source = None
    logf.close()

def main():
    
    SKELETON_DIR = r'C:\Users\DeepRS\Desktop\edge_detection\updation.py\data\dl_out_bin.H.filled.tif'

    skeleton, ds = loadGeoRaster(SKELETON_DIR, return_ds=True)

    skeleton = get_skeleton(skeleton, 1)
    
    WriteTiff(skeleton, r'C:\Users\DeepRS\Desktop\edge_detection\updation.py\data\dl_out_bin.J.skeleton.tif', ds)
    
    point_map = get_cross_end_points(skeleton)

    num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)

    M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)

    lines, nodes = get_topo_relations(M_pl)
    
    polylines = []
    indiv_lines_t   = torch.from_numpy(indivi_skeleton).cuda().float()
    indiv_points_t  = torch.from_numpy(indiv_point_map).cuda().float()

    generate_line_vec(lines, polylines, indiv_lines_t, indiv_points_t)
    
    point_to_line(polylines,  r'C:\Users\DeepRS\Desktop\edge_detection\updation.py\data\testttr.shp', ds)
    # print(polylines)
    
    # with open(r'C:\Users\DeepRS\Desktop\edge_detection\updation.py\data\line.txt', 'w') as f:

    #     for polyline in polylines:
    #         polyline = polyline.flatten()
    #         f.write(','.join([str(e) for e in polyline]))

    # thread_pool = ThreadingPool(
    #     generate_line_vec, 
    #     lines, 
    #     8, 
    #     args=[polylines, indiv_lines_t, indiv_points_t]
    #     )

    # thread_pool.run()

    # print(polylines)



# get_led_by_fmatrix()

if __name__ == '__main__':
    
    main()