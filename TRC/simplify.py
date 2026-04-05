import topojson as tp
import geopandas as gpd
# import subprocess




def topo_simplify(src_shp, dst_shp):
    
    

    gdf = gpd.read_file(src_shp)
    
    topo = tp.Topology(gdf, topology=True, prequantize=False)
    simplified = topo.toposimplify(
        epsilon=0.00002,
        simplify_algorithm='dp',
        simplify_with='shapely',
        prevent_oversimplify=False
    ).to_gdf()

    simplified.to_file(dst_shp, driver="ESRI Shapefile")
    
    return dst_shp
    # cmd = f'ogr2ogr -r "ESRI Shapefile" {dst_shp} {tmp_json}'

    

