import geopandas as gpd
import numpy as np



TARGET = r''
PRED = r''


def rule1(ir, upb = 0.5, lowb = 0.2):
    if ir > upb:
        return 1
    elif ir > lowb:
        return 2
    else:
        return 3

RULES_MAPPING = {
    1: rule1,
}

def update(target_dir, pred_dir, updated_shp, upb=0.75, lowb=0.2, forward_rule:int=1, min_island_area=200):
    
    forward_rule = RULES_MAPPING.get(forward_rule)

    target = gpd.read_file(target_dir)
    target = target.filter(['TBLXDM', 'geometry'])
    if 'FID' not in  target.columns:
        target.loc[:,'FID'] = target.index
    
    # add field
    target.loc[:, 'inter_area'] = np.zeros(len(target), dtype=np.float32)
    target.loc[:, 'inter_ratio'] = np.zeros(len(target), dtype=np.float32)
    target.loc[:, 'status'] = np.zeros(len(target), dtype=np.uint8)
    target.loc[:,'self_area'] = target.area

    pred = gpd.read_file(pred_dir)
    pred = pred.to_crs(target.crs)

    ############################################################
    ################# forward intersection #####################
    ############################################################
    forward_intersect = gpd.overlay(target, pred, how='intersection')
    forward_intersect.loc[:,'area'] = forward_intersect.area

    for _, row in forward_intersect.iterrows():
        target.at[row['FID_1'], 'inter_area'] +=  row['area']

    target.loc[:, 'inter_ratio'] = target.loc[:, 'inter_area'] / target.loc[:,'self_area']

    field_status = []
    for idx, row in target.iterrows():
        recls = forward_rule(row['inter_ratio'], upb, lowb)
        field_status.append(recls)

    target.loc[:, 'status'] = field_status


    ############################################################
    ################# backward intersection ####################
    ############################################################

    target = target[target.status < 3] # the blocks except thoes eliminated in forward pro.

    pred.loc[:,'self_area'] = pred.area
    pred = pred[pred.area>min_island_area]

    # pred.loc[:, 'inter_ratio'] = np.zeros(len(pred), dtype=np.float32)
    pred.loc[:, 'inter_area'] = np.zeros(len(pred), dtype=np.float32)


    backward_intersection = gpd.overlay(pred, target, how='intersection')
    backward_intersection.loc[:,'area'] = backward_intersection.area
    
    for _, row in backward_intersection.iterrows():
        pred.at[row['FID_1'], 'inter_area'] +=  row['area']
        
    # pred.loc[:, 'inter_ratio'] = pred.loc[:, 'inter_area'] / pred.loc[:,'self_area']
    pred = pred[pred.inter_area == 0]

    target.to_file(updated_shp, driver="ESRI Shapefile")

    return [updated_shp]
    
