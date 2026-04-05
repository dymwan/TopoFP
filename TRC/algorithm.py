# from tkinter.dialog import DIALOG_ICON
import numpy as np
import torch
from .basicTools import Information
Info = Information()

# Info = Information()

__all__ = [
    'outer_product',
    'extend_line',  
    'Point2Line',
    'getLineCrossPoint',
    'DouglasPeuker',
    'draw_line',
    'draw_line_v2',
    'build_sequential_line',
    ]  


DIST_GENERATOR = torch.nn.PairwiseDistance(p=2)



def torch_cross(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]


def outer_product(v1, v2):  
    return v1.x*v2.y - v2.x*v1.y


def extend_line(point_end, point_start, extend_length:torch.Tensor, point_final=None, line_length=None, device='cpu'):
    '''
    (point_start)x1----->x2(point end)
    x1-------(x2)---->x3 (extended point)
    '''
    LAST_TAIL_LENGTH_THRESHOLD = 12

    if point_final is not None:
        last_tail_length=  DIST_GENERATOR(
            point_final.unsqueeze(0), point_end.unsqueeze(0)
        )
    else:
        point_final = point_end
        last_tail_length = LAST_TAIL_LENGTH_THRESHOLD


    if last_tail_length > LAST_TAIL_LENGTH_THRESHOLD:
        vec = point_final -point_end
        if line_length is None:
            line_length = DIST_GENERATOR(
            point_final.unsqueeze(0), point_end.unsqueeze(0)
            )
        extend_vec = vec * (extend_length / line_length)
        return point_final + extend_vec
    else:
        vec = point_end - point_start
        if line_length is None:
            line_length = DIST_GENERATOR(
            point_end.unsqueeze(0), point_start.unsqueeze(0)
            )
        extend_vec = vec * (extend_length / line_length)
    
    extended_line = point_final+extend_vec
    return extended_line.to(device)




def Point2Line_numpy(point, line_point1, line_point2):

    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance



def getLineCrossPoint(skeleton,device='cpu'):
    '''
        get cross points and endpoints of all lines with GPU implementation
            all cross-points are labeled as 1
            all end-points are labeled as 2
    
    '''
    w, h = skeleton.shape

    skeleton_t = torch.from_numpy(skeleton).float().to(device).unsqueeze(0).unsqueeze(0)
    cross_point_map = torch.zeros(skeleton.shape, dtype=torch.uint8).to(device)

    helo_index = torch.LongTensor([[0],[1],[2],[5],[8],[7],[6],[3],[4]]).unsqueeze(0).repeat(1,1, w*h).to(device)


    shift_window = torch.nn.Unfold(3,1,1,1)
    ats = shift_window(skeleton_t)

    helos = ats.gather(1, helo_index)[:,:-1,:]

    helos -= torch.roll(helos, 1, dims=1)
    helos[helos == -1] = 0
    helos_ct =  torch.sum(helos, 1).view(*skeleton.shape)
    
    # Info(0)(cross_point_map.shape)
    # Info(0)(helos_ct.shape)
    cross_point_map[helos_ct >= 3] = 1 

    window_sum = torch.sum(ats, 1).view([skeleton_t.shape[-2],skeleton_t.shape[-1]])
    window_sum[skeleton_t[0,0,:,:]!=1] = 0

    cross_point_map[window_sum == 2] = 2
    cross_point_map[skeleton_t[0,0,:,:] == 0] = 0
    
    return cross_point_map.detach().cpu().numpy()


def get_Individual_lines(skeleton, point_map):
    ...



def Point2Line(point, line_point1, line_point2):
    # print('\n>>>>>',line_point1, line_point2)
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    

    
    distance = torch.abs(torch_cross(vec1,vec2)) / torch.linalg.norm(line_point1-line_point2)
    return distance

def DouglasPeuker(pointset, threshold, mode='removal'):
    # TODO just removal

    pointset_len = len(pointset)
    dmax = 0
    index = 0
    
    for idx, point in enumerate(pointset):
        # print(idx, point, pointset[0], pointset[-1]) #0 tensor([ 0, 40], device='cuda:0')
        # exit()
        if idx == 0 or idx == pointset_len-1: continue
        d = Point2Line(point, pointset[0], pointset[-1])
        if d>dmax:
            dmax = d
            index = idx
    
    if dmax >= threshold:
        front_half = DouglasPeuker(pointset[:index], threshold)
        latter_half= DouglasPeuker(pointset[index:], threshold)

        result_polyline = torch.cat([front_half[:-1], latter_half],axis=0)
    else:
        result_polyline = torch.stack([pointset[0], pointset[-1]],axis=0)
    
    return result_polyline

def draw_line(map, p1:torch.Tensor, p2:torch.Tensor, fill_v=1):
    
    

    n = torch.max(torch.abs(p1-p2)).long().item() + 1

    xis = torch.linspace(p1[0],p2[0], n+1).long()
    yis = torch.linspace(p1[1],p2[1], n+1).long()
    
    try:
        map[xis,yis] = fill_v
    except IndexError:
        av_idx = torch.logical_and(xis < map.shape[0]-1, yis < map.shape[1]-1)
        xis = xis[av_idx]
        yis = yis[av_idx]
        map[xis,yis] = fill_v

def draw_line_v2(map, p1:torch.Tensor, p2:torch.Tensor, fill_v=1, mode='nearest', indiv_bound_map=None, device='cuda:0'):
    '''
    p2 is the start point    
    '''
    w, h = map.size()

    n = torch.max(torch.abs(p1-p2)).long().item() + 1

    xis = torch.linspace(p1[0],p2[0], n+1).long().to(device)
    yis = torch.linspace(p1[1],p2[1], n+1).long().to(device)
    
    if_cancel_dangling = False

    if (xis.max() > w-1 or xis.min() < 0) or (yis.max() > h-1 or yis.min() < 0):
        av_idx = torch.logical_and(
            torch.logical_and(xis < w-1, xis >= 0), 
            torch.logical_and(yis < h-1, yis >= 0) )
            
        xis = xis[av_idx]
        yis = yis[av_idx]
    
    if mode.lower() == 'nearest':
        assert indiv_bound_map is not None
        touched_bound = indiv_bound_map[xis,yis]
        # if fill_v == 621:
        #     print(touched_bound, 'check', len(touched_bound))
        
        touched_bound = torch.where(
            torch.logical_and(touched_bound > 0, touched_bound != fill_v))
        touched_bound = torch.stack([xis[touched_bound], yis[touched_bound]]).permute(1,0)
        
        
        # if fill_v == 621:
        #     print(touched_bound, 'check', len(touched_bound))

        # print(touched_bound, p2, len(touched_bound))
        # print(touched_bound)
        if len(touched_bound) >= 1:
            # distance = touched_bound-p2
            # print(distance, 'check')
            #print(touched_bound.device)# cuda:0
            #print(p2.device) #cpu
            distance = torch.argmin(torch.norm(touched_bound-p2, dim=1, p=2))
            # print([xis[distance], yis[distance]])
            cloest_point= torch.cat([touched_bound[distance][0].view(-1), touched_bound[distance][1].view(-1)])
            # if fill_v == 621:
            #     print('distance', distance)
            #     print('check in, original p1', p1, cloest_point)
            #     print('indiv_bound_map[849, 1524]', indiv_bound_map[849, 1524])
            #     print('touched_bound-p2', touched_bound-p2)
            if_cancel_dangling = True
        else:
            cloest_point = p1
        
        draw_line_v2(map, cloest_point, p2, fill_v=fill_v, mode='constant', indiv_bound_map=None, device=device)

    elif mode.lower() == 'constant':
        map[xis,yis] = fill_v

    # try:
    #     map[xis,yis] = fill_v
    # except IndexError:
    #     av_idx = torch.logical_and(xis < map.shape[0]-1, yis < map.shape[1]-1)
    #     xis = xis[av_idx]
    #     yis = yis[av_idx]
    #     map[xis,yis] = fill_v
    
    return if_cancel_dangling


def build_sequential_line(sequential_set, left_chaos_set):
    # print(sequential_set[-1])
    #TODO sequential_set = build_sequential_line(node1.unsqueeze(0), b)
    # print(sequential_set.shape, left_chaos_set.shape)
    
    if len(sequential_set)>500:
        return sequential_set

    closest_idx = torch.argmin(torch.norm(left_chaos_set - sequential_set[-1], dim=1))
    

    # print(sequential_set, left_chaos_set[closest_idx,:])
    sequential_set = torch.cat([sequential_set, left_chaos_set[closest_idx,:].unsqueeze(0)])
    
    left_chaos_set = torch.cat([left_chaos_set[:closest_idx,:], left_chaos_set[closest_idx+1:,:]])
    
    # print(sequential_set)
    if len(left_chaos_set) >= 1:
        sequential_set = build_sequential_line(sequential_set, left_chaos_set)
    
    return sequential_set
