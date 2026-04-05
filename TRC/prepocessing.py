'''
This module extracts all the lines from the DL out imagery

Step.1 Extract all skeletons from the boundary area (encoding as 1,2)
Step.2 judge the boundary lines and non-boundary lines acoording the rules:
    


Line width: 
    a dict {line-id: mean-line-width}
    

'''
from operator import mod
from sre_constants import LITERAL_LOC_IGNORE
import numpy as np
import torch

from .algorithm import *
from .basicTools import *
from .utils import *
# from .Features import Line, Node
from tqdm import tqdm

# LOG = open(r'C:\Users\DeepRS\Desktop\edge_detection\updation_v1\log.txt', 'a')


cross_kernel_3 = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
], np.uint8)

# def washbound(dlout):
#     bound_bin = getIsolateCls(dlout, 1,2)
#     bound_bin = cv2.morphologyEx(bound_bin, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=1)
    
#     dlout[bound_bin == 1] = 2


# def washbound(dl_out, threshold=24):
#     bound_other_bin = getIsolateCls(dl_out, 1,2,4)
    
#     dist_map = cv2.distanceTransform(bound_other_bin, cv2.DIST_L2, 5)
    
#     num, indiv_other = cv2.connectedComponents(
#         getIsolateCls(dl_out, 4)
#     )
    
#     for i in range(num):
#         if i == 0: continue

#         dist_block = dist_map[indiv_other == i]
#         if np.max(dist_block) < threshold:
#             dl_out[indiv_other == i] = 2
#     # return dl_out



# def get_skeleton(dl_bin, *class_codes):
#     if class_codes:
#         target_bin = getIsolateCls(dl_bin, *class_codes)
#     return skeletonize(target_bin, method='lee').astype(np.uint8)

# def get_cross_end_points(skeleton):
#     point_map = getLineCrossPoint(skeleton)
    
#     return point_map


# def get_indivdual_lines(skeleton, point_map):
#     cross_point_map = point_map.copy()
#     cross_point_map[cross_point_map == 2] = 0
#     cross_point_map = cv2.dilate(cross_point_map, np.ones((3,3),np.uint8), iterations=1)
    
#     indivi_skeleton = skeleton.copy()
#     indivi_skeleton[cross_point_map == 1] = 0
#     num_lines, indivi_skeleton = cv2.connectedComponents(indivi_skeleton)
#     return num_lines, indivi_skeleton





# def get_led_by_fmatrix(indiv_skeleton, indiv_point_map, n_lines, n_points):
#     M_pl = torch.zeros((n_lines, n_points)).byte()
    
#     indiv_skeleton_t = torch.from_numpy(indiv_skeleton).float()
#     indiv_pointmap_t = torch.from_numpy(indiv_point_map).float()
#     stacked = torch.stack([indiv_skeleton_t, indiv_pointmap_t]).unsqueeze(0)

#     shift_window = torch.nn.Unfold(5,1,2,1)
#     ats = shift_window(stacked)

#     window_point = torch.where(ats[:,37,:] > 0)

#     ats_av_bound = ats[:, :25, window_point[1]].view(-1)
#     ats_av_point = ats[:, 37, window_point[1]].repeat(25,1).view(-1)

#     M_pl[tuple([ats_av_bound.long(), ats_av_point.long()])] = 1

#     return M_pl



    
# def get_lines(led_by_matrix):


#     lines = {}


#     n_lines = led_by_matrix.shape[0]
#     for line_i in range(1, n_lines):

#         points = torch.where(led_by_matrix[line_i, :] > 0)

#         if len(points[0]) < 2:
#             continue

#         coords = torch.stack([*points]).permute(1,0).tolist()
#         if len(points[0]) == 2:
#             lines[line_i] = Line(line_i, vertices= coords, linked_nodes=points[0].tolist())
#         else:
#             lines[line_i] = Line(line_i, vertices= coords, linked_nodes=points[0].tolist(), prepared=False)
    


#     return lines
        

# def get_nodes(led_by_matrix, ):
    
#     n_points=led_by_matrix.shape[1]

#     points=  {}

#     for point_i in range(1, n_points):
        
#         lines = torch.where(led_by_matrix[:, point_i] > 0)

#         nd = Node(point_i)
        
#         nd.add_linked_line(*lines[0].tolist()[1:])

#         points[point_i] = nd
    
#     return points
        
        

# def get_dual(dl_out, indiv_line, lines, nodes, threshold):
    
#     bound_bin = getIsolateCls(dl_out, 1, 2)
#     bound_dist = cv2.distanceTransform(bound_bin, cv2.DIST_L2, 5)

#     # line_codes = np.unique(indiv_line)

#     bound_dist = torch.from_numpy(bound_dist).float().cuda()
    
#     indiv_line_t = torch.from_numpy(indiv_line).float().cuda()

#     dual_line_ids = []
#     dual_point_ids = []
#     for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=100):
        
#         if lc == 0: continue
        
#         line_coords = torch.where(indiv_line_t == lc)
        
#         mean_width = torch.mean(bound_dist[line_coords])

#         lines[lc].set_length(line_coords[0].size()[0])
#         lines[lc].set_line_width(mean_width)
        
#         if mean_width*2-6 > threshold:
            
#             dilate_width = round(mean_width.item() - 8)
#             dilate_width = 1 if dilate_width < 1 else dilate_width
            
#             lines[lc].set_dual(dilate_width)
#             dual_line_ids.append(lc)

#             for node_code in lines[lc].linked_nodes:
#                 nodes[node_code].set_dual(dilate_width)
#                 dual_point_ids.append(node_code)
    
#     for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=100):
#         lines[lc].set_linked_lines(nodes)
            
#     return bound_dist # for testing


# def get_dangling_lines(lines, nodes):
    
#     dangling_line_codes = {}
    
#     for line_code in lines.keys():
        
#         node_codes = lines[line_code].linked_nodes
#         status = []
#         direction = []
#         for node_id in node_codes:
#             if len(nodes[node_id].linked_lines)  < 2:
#                 status.append(True)
#                 direction.append(node_id)
#             else:
#                 status.append(False)
        
#         if any(status) and len(status) > 0:
#             dangling_line_codes[line_code] = direction
    
       
    
#     return dangling_line_codes


# def eliminate_dangling_lines(skeleton, indiv_skeleton, dangling_lines):
#     for line_id in dangling_lines.keys():
#         skeleton[indiv_skeleton == line_id] = 0



# def get_extended_dangling_lines(dangling_lines:dict, dl_out, lines, nodes, indiv_lines, indiv_points, extend_length=60):
    
#     DANGLING_LENGTH_THRESH:int = 12
    

#     indiv_lines_t = torch.from_numpy(indiv_lines).cuda()
#     indiv_points_t= torch.from_numpy(indiv_points).cuda()


#     extended_map = torch.zeros(indiv_lines_t.shape).long().cuda()

#     # for line_code, start_point_codes in tqdm(dangling_lines.items(), ncols=100, total=len(dangling_lines)):
#     for line_code, start_point_codes in tqdm(dangling_lines.items(), total=len(dangling_lines), ncols=100, desc='extending dangling lines'):
#     # for line_code, start_point_codes in dangling_lines.items():
        
#         if lines[line_code].length < DANGLING_LENGTH_THRESH:
#             continue
            
#         line_coords = None
#         tail_point_index = None
#         subtail_point_index = None
        
#         # print(start_point_codes)
#         for point_code in start_point_codes:
#             point_coordinate = torch.where(indiv_points_t == point_code)
#             point_coordinate = torch.cat([*point_coordinate]).float()
#             # print(line_code, point_coordinate)
#             if line_coords is None:
#                 line_coords = torch.stack([*torch.where(indiv_lines_t == line_code)]).permute(1,0)
#                 line_coords = build_sequential_line(point_coordinate.unsqueeze(0), line_coords.float())
#                 line_coords = DouglasPeuker(line_coords, 2)
            
#             if lines[line_code].length / len(line_coords) < 6:
#                 continue
#             # print(point_coordinate.item(),line_coords[:,0].item())

#             if point_coordinate in line_coords[:1,:]:
#                 tail_point_index    = 0
#                 subtail_point_index = 1
#             else:
#                 tail_point_index    = -1
#                 subtail_point_index = -2
            
#             # print('>>>',tail_point_index, point_coordinate, line_coords[:,:1])
#             ep = extend_line(line_coords[tail_point_index], line_coords[subtail_point_index], extend_length)
#             draw_line_v2(extended_map, ep, line_coords[tail_point_index], fill_v = line_code, mode='nearest', indiv_bound_map=indiv_lines_t)

#     # print(failed)
#     return extended_map.cpu().int().numpy()


# def reconstruct_dlout(indiv_skeleton, indiv_node, lines):
    
    
#     dual_line_map = torch.zeros(indiv_skeleton.shape).cuda().long()

#     #TODO tmp for testing
#     line_width = torch.zeros(indiv_skeleton.shape).cuda().float()

#     extend_width = []
#     line_keys = lines.keys()
#     for line_code, lineo in lines.items():
        
#         if lineo.length < 2:
#             continue
        
#         linked_duals = [not lines[e].is_dual for e in lineo.linked_lines if e in line_keys]
#         if len(linked_duals) > 0 and all(linked_duals):
#             continue
        

#         #TODO tmp for testing
#         line_width[indiv_skeleton == line_code] = lineo.line_width

#         dual_line_map[indiv_skeleton == line_code] = lineo.expend_width
#         for node_code in lineo.linked_nodes:
#             dual_line_map[indiv_node == node_code] = lineo.expend_width

#         # dual_line_map[extended_map == line_code] = lineo.expend_width

#         extend_width.append(lineo.expend_width)

#     dual_line_map = dual_line_map.int().cpu().numpy()

#     extended_dual_line_map = np.zeros(dual_line_map.shape, dtype=np.uint8)

#     extend_width = list(set(extend_width))

#     rest_dilate_times = max(extend_width)
    
#     while rest_dilate_times > 0:
#         if rest_dilate_times not in extend_width:
#             rest_dilate_times -= 1
#         else:
        
#             print(extended_dual_line_map.shape, dual_line_map.shape, rest_dilate_times)
#             extended_dual_line_map[dual_line_map == rest_dilate_times] = 1
            
#             extended_dual_line_map = cv2.dilate(
#                 extended_dual_line_map,
#                 np.ones([3,3], dtype=np.uint8),
#                 iterations=1
#             )
            
#             rest_dilate_times -= 1

#     return dual_line_map, extended_dual_line_map, line_width
    
        
# def merge_extended_lines(skeleton,extended_line):
#     skeleton[extended_line >0] = 1
    

@timeRecord
def post_process(dl_bin, device):
        dl_bin = np.pad(dl_bin, ((1,1),(1,1)), mode='edge')

        # dl_out = laundry(dl_out)
        dl_bin = eliminate_salt_in_bound_area(dl_bin, min_isoland_area=50, device=device)

        skeleton = get_skeleton(dl_bin, 1,2)
        point_map = get_cross_end_points(skeleton)

        num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
        nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
        M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)

        lines, nodes = get_topo_relations(M_pl)
        dangling_lines = get_dangling_lines(lines, nodes)
        bound_dist = get_dual(dl_bin, indivi_skeleton, lines, nodes, 7, include_dangling=False,device=device)

        dual_line_map, extended_dual_linemap, line_width = reconstruct_dlout(indivi_skeleton, indiv_point_map, lines, nodes,device=device)
        extended_dual_linemap_over = cv2.dilate(extended_dual_linemap, np.ones([3,3],np.uint8))
        extended_dual_linemap_over[extended_dual_linemap == 1] = 0

        
        dl_bin[extended_dual_linemap == 1] = 4
        dl_bin[extended_dual_linemap_over == 1] = 2

        dl_bin[0,:] = 4
        dl_bin[-1,:] = 4
        dl_bin[:,0] = 4
        dl_bin[:,-1] = 4

        skeleton = get_skeleton(dl_bin, 1,2)
        # point_map = get_cross_end_points(skeleton)

        # num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
        # nump_point, indiv_point_map = getIndivCls(point_map, 1,2)

        # M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
        # lines, nodes = get_topo_relations(M_pl)
        # set_lines_width(dl_bin, indivi_skeleton, lines)
        # dangling_lines = get_dangling_lines(lines, nodes)

        # # most important
        
        
        # extended_map, indivi_skeleton = get_extended_dangling_lines(dangling_lines, dl_bin, lines, nodes, indivi_skeleton, indiv_point_map, extend_length=150,device=device)

        # extended_map = cv2.dilate(extended_map, kernel=np.ones([3,3], np.uint8), iterations=2)

        # skeleton[extended_map > 0] = 1
        # skeleton = eliminate_salt_in_bound_area(skeleton,device=device)
        # skeleton = get_skeleton(skeleton, 1)

        # point_map = get_cross_end_points(skeleton)
        # num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
        # nump_point, indiv_point_map = getIndivCls(point_map, 1,2)

        # M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
        # lines, nodes = get_topo_relations(M_pl)

        # dangling_lines = get_dangling_lines(lines, nodes)
        # eliminate_dangling_lines(skeleton, indivi_skeleton, dangling_lines, device=device)
        
        filled = fill_skeleton(skeleton, dl_bin, device=device)
        return filled[1:-1,1:-1]

@timeRecord
def main(dl_bin):
    # dl_bin_dir = r'C:\Users\DeepRS\Desktop\edge_detection\updation_v1\data\Merge_203817_111015_WGS84.pred.tif'
    # scale = 1
    # dl_bin, ds = loadGeoRaster(dl_bin_dir, scale=scale, return_ds=True)
    # print(dl_bin.shape)
    
    dl_bin = np.pad(dl_bin, ((1,1),(1,1)), mode='reflect')

    dl_bin = laundry(dl_bin) # (3910, 4951) -> 1.63s
    
    # WriteTiff(dl_bin, dl_bin_dir.replace('.tif', '.A.washed.tif'), ds)
    
    

    skeleton = get_skeleton(dl_bin, 1,2)
    
    point_map = get_cross_end_points(skeleton)

    num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)

    M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)

    # lines = get_lines(M_pl) #2232, 28
    # nodes = get_nodes(M_pl)
    lines, nodes = get_topo_relations(M_pl)
    
    
    bound_dist = get_dual(dl_bin, indivi_skeleton, lines, nodes, 7)

    

    # >> KEY NODE <<:reconstruct dlout to helo the dual line
    dual_line_map, extended_dual_linemap, line_width = reconstruct_dlout(indivi_skeleton, indiv_point_map, lines, nodes)
    
    extended_dual_linemap_over = cv2.dilate(extended_dual_linemap, np.ones([3,3],np.uint8))
    extended_dual_linemap_over[extended_dual_linemap == 1] = 0

    dl_bin[extended_dual_linemap == 1] = 4
    dl_bin[extended_dual_linemap_over == 1] = 2

    dl_bin[0,:] = 2
    dl_bin[-1,:] = 2
    dl_bin[:,0] = 2
    dl_bin[:,-1] = 2
    dl_bin = get_nacked_field_bound(dl_bin)
    # WriteTiff(indivi_skeleton, dl_bin_dir.replace('.tif', '.A.indivi_skeleton.tif'),ds)
    # WriteTiff(dl_bin, dl_bin_dir.replace('.tif', '.A.new_dl_bin.tif'),ds)

    skeleton = get_skeleton(dl_bin, 1,2)
    
    point_map = get_cross_end_points(skeleton)
    num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)
    # WriteTiff(indivi_skeleton, dl_bin_dir.replace('.tif', '.A1.indivi_skeleton.tif'),ds)


    M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    lines, nodes = get_topo_relations(M_pl)
    
    set_lines_width(dl_bin, indivi_skeleton, lines) #??????
    # bound_dist = get_dual(dl_bin, indivi_skeleton, lines, nodes, 7)

    dangling_lines = get_dangling_lines(lines, nodes)
    # print(dangling_lines)
    # print(dangling_lines, '>> dangling lines')
    
    extended_map, indivi_skeleton = get_extended_dangling_lines(dangling_lines, dl_bin, lines, nodes, indivi_skeleton, indiv_point_map, extend_length=150)

    



    # >> KEY NODE << merge and reconstruct skeleton
    # about 3s
    skeleton[extended_map > 0] = 1
    
    
    skeleton = get_skeleton(skeleton, 1)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, cross_kernel_3,iterations = 1)
    
    skeleton = get_skeleton(skeleton, 1)

    # WriteTiff(skeleton, dl_bin_dir.replace('.tif', '.G4.skeleton.tif'),ds)

    point_map = get_cross_end_points(skeleton)
    num_lines, indivi_skeleton = get_indivdual_lines(skeleton, point_map)
    nump_point, indiv_point_map = getIndivCls(point_map, 1,2)

    M_pl = get_led_by_fmatrix(indivi_skeleton, indiv_point_map, num_lines, nump_point)
    lines, nodes = get_topo_relations(M_pl)

    dangling_lines = get_dangling_lines(lines, nodes)
    eliminate_dangling_lines(skeleton, indivi_skeleton, dangling_lines)
    
    # WriteTiff(skeleton, dl_bin_dir.replace('.tif', '.G.skeleton.tif'),ds)
    
    # WriteTiff(dl_bin, dl_bin_dir.replace('.tif', '.H.dl_bin.tif'),ds)
    
    

    filled = fill_skeleton(skeleton, dl_bin)
    
    # if scale != 1:
    #     filled = cv2.resize(
    #                 filled, 
    #                 (0,0), 
    #                 fx=1/float(scale),
    #                 fy=1/float(scale),
    #                 interpolation=cv2.INTER_NEAREST
    #             )
    # WriteTiff(filled, dl_bin_dir.replace('.tif', '.H.filled.tif'),ds)
    
    return filled[1:-1,1:-1]

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

if __name__ == '__main__':
    # main(r'E:\pytorch_dym\test_data\predicted\test014.512.pred.tif')
    # main(r'D:\fieldExtraction\Updation\predicted\google_map.512.pred.tif')
    
    PRED_DIR = r'/home4/dymwan/test_data/result/panjing.pred.tif'
    DST__DIR = r'/home4/dymwan/test_data/result'
    
    basename = os.path.basename(PRED_DIR).split('.')[0]

    STRIDE = 5000
    SPLITING = False
    # STRIDE

    ds = gdal.Open(PRED_DIR)
    x,y = ds.RasterXSize, ds.RasterYSize
    
    if x > 5000 or y > 5000:
        SPLITING = True
        STRIDE = 5000

    raw_gt = list(ds.GetGeoTransform())
    # print(raw_gt)
    # exit()

    anchors = get_anchor(x, y, STRIDE)
    

    for xs, ys, xoff, yoff, xi, yi in anchors:
        print(xs, ys, xoff, yoff)
        # continue
        gtxy = list(ds.GetGeoTransform())

        gtxy = update_geotransform(gtxy, xs, ys)

        tmp_driver = gdal.GetDriverByName('MEM')
        # tmp_driver = gdal.GetDriverByName('GTiff')
        tmp_raster = tmp_driver.Create(
            "", xoff, yoff, 1, gdal.GDT_Int32)
        # tmp_raster = tmp_driver.Create(
        #     r'E:\pytorch_dym\test_data\predicted\final\temp_{}_{}.tif'.format(xi, yi), xoff, yoff, 1, gdal.GDT_Int32)

        tmp_raster.SetGeoTransform(gtxy)
        tmp_raster.SetProjection(ds.GetProjection())

        

        tmp_arr = ds.ReadAsArray(xs, ys, xoff, yoff)
        
        print(tmp_raster.RasterXSize, tmp_raster.RasterYSize)
        print(tmp_arr.shape)
        # continue
        
        filled = post_process(tmp_arr)
        filled = eliminate_bound(filled, )
        tmp_raster.GetRasterBand(1).WriteArray(filled.astype(np.int32))
        
        name = f'{xi}_{yi}'
        dst_shp_xy = os.path.join(DST__DIR, f'{basename}_{xi}_{yi}.shp')
        
        raster2poly(tmp_raster, dst_shp_xy, name)
        # tmp_raster.Close()
        tmp_raster.FlushCache()
        tmp_driver = None
        tmp_raster = None
        

    # exit()
    # # x_offset= 0
    # # y_offset= 0
    # i = 0
    # x_start = 0
    # while x_start < x:
    #     y_start = 0
    #     while y_start < y:
            
    #         xoffset = STRIDE if x_start + STRIDE < x else x-x_start-1
    #         yoffset = STRIDE if y_start + STRIDE < y else y-y_start-1
    #         print(yoffset)
            
            
            
            
            
            
        
            
    #         # 
            
    #         y_start += yoffset
    #         print(xoffset, yoffset,  x_start, y_start, x, y)
    #         i+= 1
    #         if i > 5:
    #             exit()


    #     x_start += xoffset 
            
    # dst_driver = gdal.GetDriverByName('GTiff')
    # outRaster = dst_driver.Create(
    #     DST__DIR, 
    #     x.shape[-1], 
    #     y.shape[-2], 
    #     1, 
    #     gdal.GDT_UInt32)

    # outRaster.SetGeoTransform(ds.GetGeoTransform())
    # outRaster.SetProjection(ds.GetProjection())

    

    

    


# LOG.close()