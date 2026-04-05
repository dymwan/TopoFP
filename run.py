import os
import json
from deep_learning_inference.inference import deep_learning_inference, get_device
from post_pro.postProcessing import run_post_pro
from post_pro.updation_geopandas import update
import logging

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(filename)
        self.handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

logger = Logger('/Share/result/log.txt')
formatter = logging.Formatter('[%(levelname)s at %(asctime)s] >> %(message)s}')
logger.handler.setFormatter(formatter)

# for linux using
INPUT = os.environ['input']
OUTPUT = os.environ['output']



# # for windows tesing
# INPUT = r'D:\fieldExtraction\Updation\input.json'
# OUTPUT= r'D:\fieldExtraction\Updation\output.json'

#with open('/Share/result/task.log', 'w') as lf:
#  lf.write('------init------\n')



with open(INPUT, 'r') as f:
    cfg = json.load(f)

# need mount
cfg["input_dir"] = cfg.get("input_dir")
cfg["dst_folder"] = cfg.get("dst_folder")


cfg["deep_model"] = cfg.get("deep_model") if os.path.isfile(cfg.get("deep_model")) \
    else "/opt/runtime/model/gaofen.tar"

cfg["target_shps"] = cfg.get("target_shps")

cfg["work_env"] = cfg.get("work_env")


'''mode ~ [1,2,3] -> [produce only, produce && updation, updation only]'''
mode = cfg.get('mode')


work_env = cfg.get("work_env")

# input_info = cfg.get("input_dir")

output_files = {"task_log": '/Share/result/log.txt'}
produced_dict = {}
updated_dict = {}

json_data = {"successful": False, "result_code":100, "data":output_files}

logger.info('parameters injected successfully')

with open(OUTPUT, 'w') as f:
    json.dump(json_data,f)



if mode in [1,2]:
    # produce only
    deep_learning_inference(cfg)
    pred_list = [
        os.path.join(work_env, e) for e in os.listdir(work_env) if e.endswith(
            cfg.get("suffix")
            )
        ]
    print('input list:')
    print(cfg.get("input_dir"))
    print('predictd list')
    print(pred_list)  
    cfg["input_dir"] = pred_list
    
    device = get_device(cfg.get("gpu_id"))
    
    produce_shps = run_post_pro(pred_list, work_root=work_env, dst_folder=cfg.get("dst_folder"), device=device, logger=logger)
    
    
    produced_dict = {"produced_{}".format(i+1): value for i, value in enumerate(produce_shps)}

    
    updated_dict = {}
    if mode == 2:
        updated_list = []
        for produce_shp in produce_shps:
            updated_shps = update(
                target_dir=cfg.get("target_shps"),
                pred_dir=produce_shp,
                updated_shp = produce_shp.replace('.shp', '.updated.shp'),
                upb=cfg.get("filter_upper_ratio"),
                lowb=cfg.get("filter_lower_ratio"),
                forward_rule=1,
                min_island_area=cfg.get("min_spot_area")
                )
            updated_list += updated_shps

        updated_dict = {"updated_{}".format(i+1): value for i, value in enumerate(updated_list)}
            
             
        

elif mode == 3:
    
    input_info = cfg.get("input_dir")
    if not input_info.endswith('.shp'):
        raise ValueError('In mode 3, the "input_dir" should be a spatial vector file, e.g. *.shp')
    
    basename = os.path.basename(input_info.replace('.shp', '.updated.shp'))
    
    
    updated_shps = update(
        target_dir=cfg.get("target_shps"),
        pred_dir=input_info,
        updated_shp = os.path.join(cfg.get("dst_folder"), basename),
        upb=cfg.get("filter_upper_ratio"),
        lowb=cfg.get("filter_lower_ratio"),
        forward_rule=1,
        min_island_area=cfg.get("min_spot_area")
        )
    updated_dict = {"updated_{}".format(i+1): value for i, value in enumerate(updated_shps)}


output_files.update(produced_dict)
output_files.update(updated_dict)


json_data = {"successful": True, "result_code":200, "data":output_files}

with open(OUTPUT, 'w') as f:
    
    json.dump(json_data,f)
    #f.close()
