import torch
from tqdm import tqdm

from .basicTools import *
from .Features import Line, Node
from skimage.morphology import skeletonize
from .algorithm import *
from osgeo import ogr, osr


# __all__ = [
#     'washbound',
#     'get_skeleton',
#     'get_cross_end_points',
#     'get_indivdual_lines',
#     'get_led_by_fmatrix',
#     # 'get_lines',
#     # 'get_nodes',
#     'get_topo_relations',
#     'get_dual',
#     'get_dangling_lines',
#     'eliminate_dangling_lines',
#     'get_extended_dangling_lines',
#     'reconstruct_dlout',
#     'merge_extended_lines',
#     'timeRecord',
#     'set_lines_width',
#     'watershed',
#     'fill_skeleton',
#     'get_nacked_field_bound',
#     'laundry',
#     'eliminate_bound',
#     'raster2poly',

# ]


KERNEL_1 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], np.uint8)

KERNEL_22 = np.array([
    [0 ,0, 1, 0, 0],
    [0 ,0, 1, 0, 0],
    [0 ,1, 1, 1, 0],
    [0 ,0, 1, 0, 0],
    [0 ,0, 1, 0, 0],
], np.uint8)
KERNEL_33 = np.array([
    [0 ,0, 0, 0, 0],
    [0 ,0, 1, 0, 0],
    [1 ,1, 1, 1, 1],
    [0 ,0, 1, 0, 0],
    [0 ,0, 0, 0, 0],
], np.uint8)
KERNEL_44 = np.array([
    [0 ,0, 0, 0, 1],
    [0 ,0, 1, 1, 0],
    [0 ,1, 1, 1, 0],
    [0 ,1, 1, 0, 0],
    [1 ,0, 0, 0, 0],
], np.uint8)
KERNEL_55 = np.array([
    [1 ,0, 0, 0, 0],
    [0 ,1, 1, 0, 0],
    [0 ,1, 1, 1, 0],
    [0 ,0, 1, 1, 0],
    [0 ,0, 0, 0, 1],
], np.uint8)
KERNEL_66 = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
    
], np.uint8)

KERNEL_2 = np.array([
    [0 ,0, 0, 1, 0, 0, 0],
    [0 ,0, 0, 1, 0, 0, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,0, 0, 1, 0, 0, 0],
    [0 ,0, 0, 1, 0, 0, 0],
], np.uint8)
KERNEL_3 = np.array([
    [0 ,0, 0, 0, 0, 0, 0],
    [0 ,0, 0, 0, 0, 0, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [1 ,1, 1, 1, 1, 1, 1],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,0, 0, 0, 0, 0, 0],
    [0 ,0, 0, 0, 0, 0, 0],
], np.uint8)
KERNEL_4 = np.array([
    [1 ,0, 0, 0, 0, 0, 0],
    [0 ,1, 1, 0, 0, 0, 0],
    [0 ,1, 1, 1, 0, 0, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,0, 0, 1, 1, 1, 0],
    [0 ,0, 0, 0, 1, 1, 0],
    [0 ,0, 0, 0, 0, 0, 1],
], np.uint8)
KERNEL_5 = np.array([
    [0 ,0, 0, 0, 0, 0, 1],
    [0 ,0, 0, 0, 1, 1, 0],
    [0 ,0, 0, 1, 1, 1, 0],
    [0 ,0, 1, 1, 1, 0, 0],
    [0 ,1, 1, 1, 0, 0, 0],
    [0 ,1, 1, 0, 0, 0, 0],
    [1 ,0, 0, 0, 0, 0, 0],
], np.uint8)


@timeRecord
def washbound(dl_out, threshold=24):
    bound_other_bin = getIsolateCls(dl_out, 1,2,4)
    
    dist_map = cv2.distanceTransform(bound_other_bin, cv2.DIST_L2, 5)
    
    num, indiv_other = cv2.connectedComponents(
        getIsolateCls(dl_out, 4)
    )
    
    for i in range(num):
        if i == 0: continue

        dist_block = dist_map[indiv_other == i]
        if np.max(dist_block) < threshold:
            dl_out[indiv_other == i] = 2
    
    bound_other_bin = getIsolateCls(dl_out, 1,2)
    bound_other_bin = cv2.morphologyEx(bound_other_bin, cv2.MORPH_CLOSE, KERNEL_22, iterations = 1)
    bound_other_bin = cv2.morphologyEx(bound_other_bin, cv2.MORPH_CLOSE, KERNEL_33, iterations = 1)
    bound_other_bin = cv2.morphologyEx(bound_other_bin, cv2.MORPH_CLOSE, KERNEL_44, iterations = 1)
    bound_other_bin = cv2.morphologyEx(bound_other_bin, cv2.MORPH_CLOSE, KERNEL_55, iterations = 1)
    bound_other_bin = cv2.morphologyEx(bound_other_bin, cv2.MORPH_CLOSE, KERNEL_66, iterations = 1)

    dl_out[bound_other_bin == 1] = 2

    dl_out[:6, :] = 2
    dl_out[-6:, :] = 2
    dl_out[:, -6:] = 2
    dl_out[:, :6] = 2
    # return dl_out

@timeRecord
def get_skeleton(dl_bin, *class_codes):
    if class_codes:
        target_bin = getIsolateCls(dl_bin, *class_codes)
    skeleton = skeletonize(target_bin, method='lee').astype(np.uint8)
    
    skeleton[1,:] = 1
    skeleton[-2,:] = 1
    skeleton[:,1] = 1
    skeleton[:,-2] = 1
    return skeleton

@timeRecord
def get_cross_end_points(skeleton):
    point_map = getLineCrossPoint(skeleton)
    
    return point_map


@timeRecord
def get_indivdual_lines(skeleton, point_map):
    cross_point_map = point_map.copy()
    cross_point_map[cross_point_map == 2] = 0
    cross_point_map = cv2.dilate(cross_point_map, np.ones((3,3),np.uint8), iterations=1)
    
    indivi_skeleton = skeleton.copy()
    indivi_skeleton[cross_point_map == 1] = 0
    num_lines, indivi_skeleton = cv2.connectedComponents(indivi_skeleton)
    return num_lines, indivi_skeleton

@timeRecord
def get_led_by_fmatrix(indiv_skeleton, indiv_point_map, n_lines, n_points):
    M_pl = torch.zeros((n_lines, n_points)).byte()
    
    indiv_skeleton_t = torch.from_numpy(indiv_skeleton).float()
    indiv_pointmap_t = torch.from_numpy(indiv_point_map).float()
    stacked = torch.stack([indiv_skeleton_t, indiv_pointmap_t]).unsqueeze(0)

    shift_window = torch.nn.Unfold(5,1,2,1)
    ats = shift_window(stacked)

    window_point = torch.where(ats[:,37,:] > 0)

    ats_av_bound = ats[:, :25, window_point[1]].view(-1)
    ats_av_point = ats[:, 37, window_point[1]].repeat(25,1).view(-1)

    M_pl[tuple([ats_av_bound.long(), ats_av_point.long()])] = 1

    return M_pl


@timeRecord   
def get_lines(led_by_matrix):


    lines = {}


    n_lines = led_by_matrix.shape[0]
    for line_i in range(1, n_lines):

        points = torch.where(led_by_matrix[line_i, :] > 0)

        if len(points[0]) < 2:
            continue

        coords = torch.stack([*points]).permute(1,0).tolist()
        if len(points[0]) == 2:
            lines[line_i] = Line(line_i, vertices= coords, linked_nodes=points[0].tolist())
        else:
            lines[line_i] = Line(line_i, vertices= coords, linked_nodes=points[0].tolist(), prepared=False)

        # if line_i in [1114, 1117, 1119]:
        #     print(line_i, '>>', points[0].tolist())
    


    return lines
        
@timeRecord
def get_nodes(led_by_matrix, ):
    
    n_points=led_by_matrix.shape[1]

    points=  {}

    for point_i in range(1, n_points):
        
        lines = torch.where(led_by_matrix[:, point_i] > 0)

        nd = Node(point_i)
        
        nd.add_linked_line(*lines[0].tolist()[1:])

        points[point_i] = nd
        # if point_i == 1007:
        #     print(point_i, '>>>>', lines[0].tolist())
    return points

@timeRecord
def get_topo_relations(led_by_matrix):
    lines = get_lines(led_by_matrix)
    nodes = get_nodes(led_by_matrix)
    for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=50):
        lines[lc].set_linked_lines(nodes)
    
    return lines, nodes

@timeRecord
def set_lines_width(dl_out, indiv_line, lines, device='cpu'):
    bound_bin = getIsolateCls(dl_out, 1, 2)
    bound_dist = cv2.distanceTransform(bound_bin, cv2.DIST_L2, 5)

    # line_codes = np.unique(indiv_line)

    bound_dist = torch.from_numpy(bound_dist).float().to(device)
    
    indiv_line_t = torch.from_numpy(indiv_line).float().to(device)

    for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=50):
        
        if lc == 0: continue
        
        line_coords = torch.where(indiv_line_t == lc)
        
        mean_width = torch.mean(bound_dist[line_coords])
        minwidth = torch.min(bound_dist[line_coords])

        lines[lc].set_length(line_coords[0].size()[0])
        lines[lc].set_line_width(mean_width, minwidth)


@timeRecord
def get_dual(dl_out, indiv_line, lines, nodes, threshold, include_dangling=True, device='cpu'):
    '''
    get the width of each individual_lines, and marked the dual-line.
    markd eaches nodes with the largest width from the linked lines
    '''    
    bound_bin = getIsolateCls(dl_out, 1, 2)
    bound_dist = cv2.distanceTransform(bound_bin, cv2.DIST_L2, 5)

    # line_codes = np.unique(indiv_line)

    bound_dist = torch.from_numpy(bound_dist).float().to(device)
    
    indiv_line_t = torch.from_numpy(indiv_line).float().to(device)

    dual_line_ids = []
    dual_point_ids = []
    for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=50):
        
        if lc == 0: 
            continue
        if not include_dangling and lines[lc].is_dangling:
            continue
        
        line_coords = torch.where(indiv_line_t == lc)
        
        mean_width = torch.mean(bound_dist[line_coords])
        minwidth = torch.min(bound_dist[line_coords])

        lines[lc].set_length(line_coords[0].size()[0])
        lines[lc].set_line_width(mean_width, minwidth)
        
        if mean_width*2-8 > threshold:
            # TODO there should be 
            
            dilate_width = round(mean_width.item() - 12)
            dilate_width = 1 if dilate_width < 1 else dilate_width
            
            lines[lc].set_dual(dilate_width)
            dual_line_ids.append(lc)

            for node_code in lines[lc].linked_nodes:
                nodes[node_code].set_dual(dilate_width)
                dual_point_ids.append(node_code)
    
    

    # set the lines that between 2 dual lines as dual
    # and the short dual lines between lines that are all not dual are set to not dual
    for lc in tqdm(lines.keys(), total=len(lines.keys()), ncols=50):
        

        if lines[lc].is_dual and lines[lc].length > 12:
            continue
        
        
        # if lines[lc].is_dual and lines[lc].length <= 12:
        #     print(lc, lines[lc].length, lines[lc].linked_nodes)
        try:
            link_dual = [lines[e].is_dual for e in lines[lc].linked_lines]
        except:
            continue
        n_linked_duals = np.bincount(link_dual, minlength=2)
        
        if n_linked_duals[1] >= 2 and lines[lc].min_line_width > 3:
            lines[lc].set_dual(1)
        elif lines[lc].is_dual and n_linked_duals[1] == 0:
            lines[lc].set_dual(1, is_dual = False)
        
        
        
        
    
    return bound_dist # for testing

def eliminate_salt_in_bound_area(im, min_isoland_area=50, device='cpu'):
    n_indiv_nb, indiv_non_bound, stats_nb, _ = getIndivClsWithStats(im, 0,3,4, connectivity=4)
    indiv_non_bound = torch.from_numpy(indiv_non_bound).to(device)
    stats_nb = torch.Tensor(stats_nb[:,-1]).to(device)
    im = torch.from_numpy(im).to(device)
    
    for bi, area in enumerate(stats_nb):
        if area >= min_isoland_area:
            continue
        im[indiv_non_bound == bi] = 2

    return im.cpu().numpy()

    
    


@timeRecord
def get_dangling_lines(lines, nodes):
    # this place can be modified via the type of different points
    dangling_line_codes = {}
    
    for line_code in lines.keys():
        
        node_codes = lines[line_code].linked_nodes
        status = []
        direction = []
        for node_id in node_codes:
            if len(nodes[node_id].linked_lines)  < 2:
                status.append(True)
                direction.append(node_id)
            else:
                status.append(False)
        
        if any(status) and len(status) > 0:
            dangling_line_codes[line_code] = direction
            lines[line_code].set_dangling()
        
        # if line_code == 1287:
        #     print(node_codes)
        #     print(status)
        #     print(direction)
        #     print(lines[1287].set_dangling())
       
    
    return dangling_line_codes

@timeRecord
def eliminate_dangling_lines(skeleton, indiv_skeleton, dangling_lines, device='cpu'):
    skeleton_t = torch.from_numpy(skeleton).to(device)
    indiv_skeleton_t = torch.from_numpy(indiv_skeleton).to(device)

    for line_id in dangling_lines.keys():
        skeleton_t[indiv_skeleton_t == line_id] = 0
    skeleton = skeleton_t.cpu().numpy()


@timeRecord
def laundry(im, min_island_area=100, device='cpu'):
    n_indiv_nb, indiv_non_bound, stats_nb, _ = getIndivClsWithStats(im, 0, 3, 4)
    n_indiv_field, indiv_field, stats_field, _ = getIndivClsWithStats(im, 0,1,2,4)
    n_indiv_other, indiv_other, stats_other, _ = getIndivClsWithStats(im, 0,1,2,3)
    
    indiv_non_bound = torch.from_numpy(indiv_non_bound).to(device)
    indiv_field = torch.from_numpy(indiv_field).to(device)
    indiv_other = torch.from_numpy(indiv_other).to(device)

    stats_nb = torch.Tensor(stats_nb[:,-1]).to(device)
    stats_field = torch.Tensor(stats_field[:,-1]).to(device)
    stats_other = torch.Tensor(stats_other[:,-1]).to(device)
    
    im = torch.from_numpy(im).to(device)

    
    for i in range(n_indiv_nb):
        if i == 0:
            continue

        fields_in_block = torch.unique(indiv_field[indiv_non_bound == i])
        others_in_block = torch.unique(indiv_other[indiv_non_bound == i])
        print(fields_in_block)
        print(others_in_block)

        # if fields_in_block.size(dim=0)+others_in_block.size(dim=0) >= 1:
            
            

        for fi in fields_in_block:
            if stats_field[fi] < min_island_area:
                im[indiv_field == fi] = 3
        for fi in others_in_block:
            if stats_other[fi]< min_island_area:
                im[indiv_other == fi] = 4
                
    return im.cpu().numpy()
    
            
            
            

        

        

    
    
    

@timeRecord
def laundry_v1(im, min_island_area = 100, device='cpu'):

    non_bound_bin = getIsolateCls(im, 0, 3, 4)
    n_indiv, indiv_non_bound, stats, centroids = cv2.connectedComponentsWithStats(non_bound_bin)
    
    indiv_non_bound = torch.from_numpy(indiv_non_bound).to(device)
    im_t = torch.from_numpy(im).to(device)
    stats = torch.Tensor(stats[:,-1]).to(device)
    
    

    for i in range(n_indiv):
        if i == 0:
            continue
        
        area = stats[i]

        contains = im_t[indiv_non_bound == i]
        uniq_values = torch.unique(contains)
        if uniq_values.size == 1:
            continue
        else:
            counts = torch.bincount(contains, minlength=5)
            
            # counts[counts < min_island_area] = 0
            
            field_area = counts[3]
            non_field_area = counts[4]
            # print(area)
            # print(field_area)
            if field_area / area < 0.1: 
                if field_area < min_island_area:
                    im_t[indiv_non_bound == i] = 4

    im = im_t.cpu().numpy()
    # print(im.shape, 'after')
    im = get_nacked_field_bound(im)
    
    return im

def get_nacked_field_bound(im):
    field = getIsolateCls(im, 3)
    dilate_field = cv2.dilate(field, np.ones([3,3], np.uint8))
    nacked_area = np.logical_and(dilate_field == 1, im==4)
    
    im[nacked_area] = 2
    return im


@timeRecord
def get_extended_dangling_lines(dangling_lines:dict, dl_out, lines, nodes, indiv_lines, indiv_points, extend_length=70, device='cpu'):
    
    DANGLING_LENGTH_THRESH:int = 12
    

    indiv_lines_t = torch.from_numpy(indiv_lines).to(device).float()
    indiv_points_t= torch.from_numpy(indiv_points).to(device).float()


    extended_map = torch.zeros(indiv_lines_t.shape).to(device).float()
    
    # eliminate the short lines
    for line_code, start_point_codes in dangling_lines.items():
        if lines[line_code].length <= DANGLING_LENGTH_THRESH:
            indiv_lines_t[torch.where(indiv_lines_t == line_code)] = -1

    for line_code, start_point_codes in dangling_lines.items():
        
        if lines[line_code].length <= DANGLING_LENGTH_THRESH:
            continue
            
        line_coords = None
        tail_point_index = None
        subtail_point_index = None
        
        extended_map[torch.where(indiv_lines_t == line_code)] = line_code



        touched = []
        # print(start_point_codes)
        for point_code in start_point_codes:

            point_coordinate = torch.where(indiv_points_t == point_code)
            
            point_coordinate = torch.cat([*point_coordinate]).float()
            
            if len(point_coordinate) > 2:
                continue

            # print(line_code, point_coordinate)
            if line_coords is None:
                line_coords = torch.stack([*torch.where(indiv_lines_t == line_code)]).permute(1,0)
                # print(line_code)
                line_coords = build_sequential_line(point_coordinate.unsqueeze(0), line_coords.float())
                

                line_coords = DouglasPeuker(line_coords, 2)
            
            if lines[line_code].length / len(line_coords) < 6:
                continue
            # print(point_coordinate.item(),line_coords[:,0].item())





            n_points, _ = line_coords.shape
            

            
            # if point_coordinate in line_coords[:1,:]:
            #     if n_points > 2:
            #         tail_point_index    = 1
            #         subtail_point_index = 2
            #         final_point_index = 0
            #     else:
            #         tail_point_index    = 0
            #         subtail_point_index = 1
            #         final_point_index = 0
            # else:
            #     if n_points > 2:
            #         tail_point_index    = -2
            #         subtail_point_index = -3
            #         final_point_index = -1
            #     else:
            #         tail_point_index    = -1
            #         subtail_point_index = -2
            #         final_point_index = -1
            
            if point_coordinate in line_coords[:1, :]:
                if n_points > 2:
                    point_end = line_coords[1]
                    point_start = line_coords[2]
                    point_final = line_coords[0]
                else:
                    point_end = line_coords[0]
                    point_start = line_coords[1]
                    point_final = None
            else:
                if n_points > 2:
                    point_end = line_coords[-2]
                    point_start = line_coords[-3]
                    point_final = line_coords[-1]
                else:
                    point_end = line_coords[-1]
                    point_start = line_coords[-2]
                    point_final = None


            ## test
            # if point_final is not None:
            #     print(line_code, point_code)


            ep = extend_line(
                point_end, 
                point_start, 
                extend_length, 
                point_final,
                device=device
                )

            draw_start = point_end if point_final is None else point_final
            #print(indiv_lines_t.device)
            #exit()
            if_cancel_dangling = draw_line_v2(
                extended_map, 
                ep, 
                draw_start, 
                fill_v = line_code, 
                mode='nearest', 
                indiv_bound_map=indiv_lines_t,
                device=device
                )
            
            touched.append(if_cancel_dangling)
        
        if all(touched):
            lines[line_code].cancel_dangling()

    extended_map = extended_map.cpu().int().numpy()

    # short_lines = np.where(extended_map == -1)
    extend_lines = np.where(extended_map > 0)
    
    extended_map_bin = np.zeros(indiv_lines_t.shape, np.uint8)
    extended_map_bin[extend_lines] = 1

    # kernel = np.array([
    #     [0,0,0,1,0,0,0],
    #     [0,0,1,1,1,0,0],
    #     [0,1,1,1,1,1,0],
    #     [1,1,1,1,1,1,1],
    #     [0,1,1,1,1,1,0],
    #     [0,0,1,1,1,0,0],
    #     [0,0,0,1,0,0,0],
        
    # ], np.uint8)

    # extended_map_bin = cv2.morphologyEx(extended_map_bin, cv2.MORPH_CLOSE, kernel,iterations = 1)
    
    # # extended_map_bin = get_skeleton(extended_map_bin, 1)
    # # extended_map_bin[short_lines] = 2
    
    # indiv_lines_t += 1
    return extended_map_bin, indiv_lines_t.long().cpu().numpy().astype(np.uint8)

@timeRecord
def reconstruct_dlout(indiv_skeleton, indiv_node, lines, nodes, device='cpu'):
    
    indiv_skeleton_t = torch.from_numpy(indiv_skeleton).to(device)
    indiv_node_t = torch.from_numpy(indiv_node).to(device)

    dual_line_map = torch.zeros(indiv_skeleton_t.shape).to(device).long()

    #TODO tmp for testing
    line_width = torch.zeros(indiv_skeleton_t.shape).to(device).float()

    extend_width = []
    line_keys = lines.keys()
    # for line_code, lineo in tqdm(lines.items(), total=len(lines), ncols=100):
    for line_code, lineo in tqdm(lines.items(), total=len(lines), ncols=100, desc="reconstructing DL out"):

        if not lineo.is_dual:
            continue
        #TODO tmp for testing
        line_width[indiv_skeleton_t == line_code] = lineo.line_width

        dual_line_map[indiv_skeleton_t == line_code] = lineo.expend_width
        for node_code in lineo.linked_nodes:
            # print(line_code, node_code)
            dual_line_map[indiv_node_t == node_code] = lineo.expend_width

        # dual_line_map[extended_map == line_code] = lineo.expend_width

        extend_width.append(lineo.expend_width)



    dual_line_map = dual_line_map.int().cpu().numpy()

    extended_dual_line_map = np.zeros(dual_line_map.shape, dtype=np.uint8)

    extend_width = list(set(extend_width))
    extend_width = [e for e in extend_width if e > 0]

    # print(extend_width)

    if len(extend_width)==0:
        return dual_line_map, extended_dual_line_map, line_width
    

    rest_dilate_times = max(extend_width)
    
    extended_dual_line_map[dual_line_map == rest_dilate_times] = 1
    while rest_dilate_times > 0:
        if rest_dilate_times not in extend_width:
            rest_dilate_times -= 1
        else:
    
            extended_dual_line_map = cv2.dilate(
                extended_dual_line_map,
                np.ones([3,3], dtype=np.uint8),
                iterations=1
            )
            
            rest_dilate_times -= 1
            if rest_dilate_times > 0:
                extended_dual_line_map[dual_line_map == rest_dilate_times] = 1

    return dual_line_map, extended_dual_line_map, line_width
    
@timeRecord        
def merge_extended_lines(skeleton,extended_line):
    skeleton[extended_line > 0] = 1
    
@timeRecord
def reconstruct_dlout1(extended_map, lines, dl_out):
    ...

@timeRecord
def watershed(skeleton, dl_out, device='cpu'):
    # st = cv2.dilate(skeleton, np.ones([5,5], np.uint8))
    st = skeleton + 1
    st[st !=  1] = 0
    
    n_field, iso_field = cv2.connectedComponents(st, connectivity=4)
    iso_field = torch.from_numpy(iso_field).to(device).long()
    dl_out_t = torch.from_numpy(dl_out).to(device).long()
    seed = torch.zeros(dl_out.shape).to(device).long()
    
    for i in range(n_field):
        if i == 0:
            continue
        
        block = dl_out_t[iso_field == i].view(-1)
        block = torch.bincount(block)
        seed[iso_field == i] = torch.argmax(block)
    print(n_field)
    # return st.cpu().int().numpy()
    return seed.cpu().int().numpy()

@timeRecord        
def fill_skeleton(skeleton, dlout, min_isoland_keep = 167, device='cpu'):

    

    dlout = get_nacked_field_bound(dlout)
    skeleton_t = torch.from_numpy(skeleton).to(device).long()

    skeleton += 1
    skeleton[skeleton == 2] = 0
    
    n_field, indiv_field = cv2.connectedComponents(skeleton, connectivity=4)

    indiv_field = torch.from_numpy(indiv_field).to(device).float()
    dlout_t = torch.from_numpy(dlout).to(device).long()
    
    
    for fi in range(1, n_field):
        blockid = torch.where(indiv_field == fi)
        
        block_raw = dlout_t[blockid]
        # print(block_raw)
        bin_count = torch.bincount(block_raw, minlength=4)
        bin_count[2] = 0
        filling_number = torch.argmax(bin_count)
        skeleton_t[blockid] = filling_number

        if filling_number == 4 and bin_count[3] > min_isoland_keep:
            skeleton_t[blockid] = dlout_t[blockid]
    


    skeleton_t = skeleton_t.cpu().numpy().astype(np.uint8)
    
    skeleton_t[skeleton_t == 2] = 4

    skeleton_t[skeleton_t > 1] -= 1
    
    return skeleton_t

def eliminate_bound(filled, bound_code=-1):
    '''
    filled encodings:
        {
            1: bound
            2: field
            3: non-field
        }
    '''
    filled_filed = getIsolateCls(filled, 2)
    n, filled_filed = cv2.connectedComponents(filled_filed, connectivity=4)
    
    

    filled_filed += 4
    filled_filed[filled_filed==4] = 0

    filled[filled == 2] = filled_filed[filled == 2]
    
    # global ds
    # WriteTiff(
    #     filled_filed, 
    #     r'E:\pytorch_dym\test_data\predicted\test014.512.pred.TTT.tif',
    #     ds
    # )
    
    
    filled = torch.from_numpy(filled).float().unsqueeze(0).unsqueeze(0)
    
    filled[filled == 3] = 0

    shift_window = torch.nn.Unfold(3,1,1,1)
    ats = shift_window(filled) #  torch.Size([1, 9, 19362320])


    non_bound_points = torch.where(ats[:,4,:] != 1) 

    ats_max_v, ats_max_idx = torch.max(ats, dim=1) # torch.Size([1, 19362320])
    
    ats_max_v[non_bound_points] = 0
    
    ats_max_v = ats_max_v.view(*filled.shape[-2:])

    filled[:,:,ats_max_v > 0] = ats_max_v[ats_max_v>0]
    
    # ats
    print(filled.shape)
    filled[filled == 3] = 0
    filled[filled == 1] = 0
    return filled.squeeze(0).squeeze(0).cpu().numpy()


def raster2poly(inraster, outshp, name='layer'):
    if isinstance(inraster, str):
        inraster = gdal.Open(inraster)  
    inband = inraster.GetRasterBand(1)  
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  
    print(prj)
 
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  
    Poly_layer = Polygon.CreateLayer(name, srs=prj, geom_type=ogr.wkbMultiPolygon)  
    newField = ogr.FieldDefn('value', ogr.OFTReal)  
    Poly_layer.CreateField(newField)
 
    gdal.Polygonize(inband, None, Poly_layer, 0)
    # gdal.FPolygonize(inband, None, Poly_layer, 0)
    
    
    strFilter = "value = '" + str(0) + "'"
    Poly_layer.SetAttributeFilter(strFilter)
    for pFeature in Poly_layer:
        pFeatureFID = pFeature.GetFID()
        Poly_layer.DeleteFeature(int(pFeatureFID))


    Polygon.SyncToDisk()
    Polygon = None



if __name__ == '__main__':

    # filled_test_dir = r'D:\fieldExtraction\Updation\predicted\temp_1_1.tif'
    # filled, ds = loadGeoRaster(filled_test_dir, return_ds=True)


    # filled = eliminate_bound(filled, )
    # WriteTiff(
    #     filled.cpu().numpy(),
    #     r'D:\fieldExtraction\Updation\predicted\temp_1_1.tmp.tif',
    #     ds=ds
    #      )
    
    raster2poly(r'D:\fieldExtraction\Updation\predicted\temp_1_1.tif', r'D:\fieldExtraction\Updation\predicted\temp_1_1.tmp.shp')
    
