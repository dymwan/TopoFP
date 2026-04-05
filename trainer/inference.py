from tqdm import tqdm
from osgeo import gdal
import numpy as np
import torch
import json
import os,sys,shutil
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

sys.path.append(r'/opt/runtime/src')
# import encoding

# # from inference_config import _C as cfg
# from .inference_utils import loadGeoRaster, get_device, build_folder, get_filelist, WriteTiff
# # from .inference_modeldef import get_model


import argparse
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

def get_shifting_steps(l, s):
    return l // s if l % s == 1 else l // s + 1

def nagetive_pad_check(start, bs):
    if start < 0:
        neg_pad = -start
        start = 0
        out_start = 0
    else:
        neg_pad = 0
        out_start = start + bs
    return neg_pad, start, out_start

def positive_pad_check(start, ps , r, bs, stride, neg_pad):

    if start + ps < r:
        offset = ps
        out_offset = stride
        pos_pad = 0
    else:
        offset = r - start
        pos_pad = ps + start - r
        if offset < stride + bs:
            out_offset = offset - bs
        else:
            out_offset = stride
    
    if neg_pad > 0:
        offset -= neg_pad

    return pos_pad, offset, out_offset
        
class inference_loader:
    
    def __init__(
                    self, 
                    src_dir, 
                    dst_dir, 
                    patch_size=256, 
                    buffer_size=32, 
                    out_channel=1, 
                    odtype=gdal.GDT_Byte, 
                    overwrite_out=False,
                    load_approach=1,
                ) -> None:

        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self.ps = patch_size
        self.bs = buffer_size
        
        self.stride = self.ps - 2* self.bs
        if self.stride <= 0:
            raise ValueError('buffersize should lt ps/2')

        self.ds = gdal.Open(self.src_dir)
        
        self.rx = self.ds.RasterXSize
        self.ry = self.ds.RasterYSize

        self.clip_map = []
        self.ods = None
        self.out_channels = out_channel
        self.out_dtype = odtype
        self.overwrite_out = overwrite_out
        
        if self.out_channels == 1:
            self.write_fun = self.save_patch2d
        else:
            self.write_fun = self.save_patch3d
        
        self.station= 0

        self.approach = load_approach

        self.get_shifting_map()
    
    def Create_out_file(self):
        driver = gdal.GetDriverByName('GTiff')
        
        self.ods = driver.Create(
            self.dst_dir, self.rx, self.ry, self.out_channels, self.out_dtype
        )
        
        self.ods.SetProjection(self.ds.GetProjection())
        self.ods.SetGeoTransform(self.ds.GetGeoTransform())

    def get_shifting_map(self):

        
        nx = get_shifting_steps(self.rx, self.stride)
        ny = get_shifting_steps(self.ry, self.stride)
        
        for xi in range(nx):
            for yi in range(ny):

                xs = xi * self.stride - self.bs
                ys = yi * self.stride - self.bs
                
                lpadx, xs, oxs = nagetive_pad_check(xs, self.bs)
                upady, ys, oys = nagetive_pad_check(ys, self.bs)
                
                
                rpadx, xoff, oxoff = positive_pad_check(
                    xs, self.ps, self.rx, self.bs, self.stride, lpadx)
                bpady, yoff, oyoff = positive_pad_check(
                    ys, self.ps, self.ry, self.bs, self.stride, upady)
                
                self.clip_map.append(
                    [xs, ys, xoff, yoff, oxs, oys, oxoff, oyoff, \
                     lpadx, rpadx, upady, bpady]
                )
    
    @staticmethod
    def data_processor(sub_patch:np.ndarray, approach=2) -> torch.Tensor:
        assert approach in [1,2,3]
        if approach == 1:
            sub_patch = torch.from_numpy(sub_patch).float().unsqueeze(dim=0)
        elif approach == 2:
            sub_patch = Image.fromarray(sub_patch.astype(np.uint8).transpose(1,2,0))#.convert('RGB')
            sub_patch = transforms.ToTensor()(sub_patch).float().unsqueeze(dim=0)
        elif approach ==3:
            sub_patch = Image.fromarray(sub_patch.astype(np.uint8).transpose(1,2,0))#.convert('RGB')
            sub_patch = transforms.ToTensor()(sub_patch).float().unsqueeze(dim=0)
            #sub_patch = transforms.Normalize([0.2638, 0.3007, 0.2999],[0.1292, 0.1126, 0.1007])(sub_patch)
            sub_patch = transforms.Normalize([0.3722, 0.4498, 0.3964],[0.1220, 0.1139, 0.1064])(sub_patch)
        else:
            raise
        return sub_patch

    def __len__(self):
        return len(self.clip_map)
    
    def __getitem__(self, index):
        self.station = index

        cm = self.clip_map[index]
        get_params = cm[:4]
        lpadx, rpadx, upady, bpady = cm[-4:]
        patch_arr = self.ds.ReadAsArray(*get_params)
        w1, h1 = patch_arr.shape[-2:]
        
        if not all([lpadx, rpadx, upady, bpady]):
            if len(patch_arr.shape) == 2:
                patch_arr = np.pad(
                    patch_arr, 
                    ((upady, bpady), (lpadx,rpadx), ), 
                    'reflect')
            else:
                patch_arr = np.pad(
                    patch_arr, 
                    ((0,0), (upady, bpady), (lpadx,rpadx), ), 
                    'reflect')
        
        patch_proc = self.data_processor(patch_arr, approach=self.approach)
        

        return index, patch_proc, torch.from_numpy(patch_arr).to(torch.int64)
                
    def save_patch(self, saving_patch:np.ndarray, index=None):
        if self.ods is None:
            self.Create_out_file()

        index = self.station if index is None else index
        
        self.write_fun(saving_patch, index=index)
        
    
    def save_patch2d(self, save_patch:np.ndarray, index=None):
        
        index = self.station if index is None else index

        cm = self.clip_map[index]
        oxoff, oyoff, oxsize, oysize = cm[4:8]
        lpadx, rpadx, upady, bpady = cm[-4:]
        
        if all([lpadx, rpadx, upady, bpady]):
            save_patch = save_patch[self.bs:-self.bs, self.bs:-self.bs]
        else:
            clipxs = lpadx if lpadx >= self.bs else self.bs
            clipys = upady if upady >= self.bs else self.bs

            clipxe = clipxs + oxsize
            clipye = clipys + oysize


            save_patch = save_patch[ clipys: clipye, clipxs: clipxe, ]
        
        band = self.ods.GetRasterBand(1)
        #self.ods.GetRasterBand(1).WriteArray(save_patch, oxoff, oyoff)
        band.WriteArray(save_patch, oxoff, oyoff)
        band.FlushCache()
        
    def save_patch3d(self, save_patch:np.ndarray, index):
        pass
            

class prefetcher:
    def __init__(self, loader) -> None:
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()


    def preload(self):
        try:
            self.next_station, self.next_patch, self.next_raw_patch = next(self.loader)
        except StopIteration:
            self.next_patch = None
            self.next_station = None
            self.next_raw_patch = None
            return 

        with torch.cuda.stream(self.stream):
            self.next_patch = self.next_patch.cuda(non_blocking=True) # cuda(non_blocking=True)
            self.next_raw_patch = self.next_raw_patch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_patch = self.next_patch
        station = self.next_station
        raw_patch = self.next_raw_patch
        
        self.preload()
        
        return station, next_patch, raw_patch

def get_file_list(input_info):
    file_list = []
    
    if isinstance(input_info, str):
        file_list = [input_info]
    elif isinstance(input_info, list):
        file_list = input_info
    return file_list

def build_folder(*folders, **kwargs):
    if 'overwrite' in kwargs:
        overwrite = kwargs['overwrite']
    else:
        overwrite = False

    for folder in folders:
        if not overwrite:
            os.makedirs(folder, exist_ok=True)
        else:
            try:
                os.makedirs(folder, exist_ok=False)
            except OSError:
                shutil.rmtree(folder)
                os.makedirs(folder)

def frozen_bn(layer):
    if isinstance(layer, torch.nn.BatchNorm2d):
        layer.track_running_stats=False # model.eval












def deep_learning_inference(cfg, logger):
    
    device = get_device(cfg.get("gpu_id"))
    logger.info('use device: %s' % device)

    src_list = get_file_list(cfg.get('input_dir'))
    dst_folder = cfg.get("work_env")

    

    build_folder(dst_folder, overwrite=cfg.get("overwrite"))
    
    
    # print(f"there are(is) {len(src_list)} images to process:")

    # print(cfg.get("deep_model"))
    
    loaded_pretrained = torch.load(cfg.get("deep_model"), map_location=device)
    # print(loaded_pretrained)
    logger.info('using deep model at %s' % cfg.get("deep_model"))
    logger.info('using data loading approach: %s' % cfg.get("loading_approach"))


    net = loaded_pretrained['model']
    if hasattr(net, 'module'):
        net = net.module
    net = net.to(device=device)
    net.eval()
    net.apply(frozen_bn)




    for idx, file_dir in enumerate(src_list):
        
        basename = os.path.basename(file_dir)

        dst_dir = os.path.join(dst_folder, basename)
        
        print(cfg.get('overwrite'))
        print(dst_dir)
        print(os.path.isfile(dst_dir))
        if not cfg.get("overwrite") and os.path.isfile(dst_dir):
            print(f'skipping {dst_dir}')
            continue
        
        loader = inference_loader(
                    src_dir=file_dir, 
                    dst_dir=dst_dir, 
                    patch_size=cfg.get('patch_size'), 
                    buffer_size=cfg.get('buffersize'), 
                    out_channel=1,
                    odtype=gdal.GDT_Byte, 
                    overwrite_out=False,
                    load_approach=cfg.get('loading_approach'),
        )

        
        data_fetcher = prefetcher(loader)
        
        npatches = len(loader)
        tbar = tqdm(range(npatches), total=npatches, ncols=80, desc='patchly inferencing')
        
        for i in tbar:
            
            station, patch, raw_patch = data_fetcher.next()
            
            # back = torch.where(patch[0,0,:,:] == 0)
            
            # print(back.shape)

            with torch.no_grad():
                pred_patch = net(patch)

            if isinstance(pred_patch, tuple):
                pred_patch = pred_patch[0]

            probs = F.softmax(pred_patch, dim=1)
            
            _, pred_patch = torch.max(probs, dim=1)
            # print(pred_patch.shape) #[1, 512, 512]
            #atch = raw_patch.to(torch.int64)
            pred_patch = torch.where(raw_patch[0,:,:] == 0, raw_patch[0,:,:], pred_patch)

            probs_cpu = pred_patch.cpu().squeeze(0).detach().numpy()


            loader.save_patch(probs_cpu, station)

            # pred_patch = np.argmax(probs_cpu, axis=0)
        logger.info('deep learning inference successfully done.')
        loader.ods.FlushCache()
        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    # test_im = r'Z:\yangben\GaofenII_Zhejiang\data1st\image\ELDOM330502.tif'
    
    # o = inference_loader(test_im, r'D:\CFP-master\test\reslt.tif', 1024, 32, 1)
    # from tqdm import tqdm
    # tbar = tqdm(o, total=len(o), ncols=50, desc=f'up to {o.station}')
    # for patch in tbar:
        
    #     # save_patch = np.ones(patch.shape[-2:], dtype=np.uint8)
    #     # save_patch *= o.station
    #     # # save_patch[...] = o.station
        
    #     patch = patch[0,:,:]

    #     o.save_patch(patch)

    #     tbar.desc = f'up to {o.station}'
        
    args = getArgs()
    if not os.path.isfile(args.cfg):
        raise Exception('Can not load inference configuration: [%s]' % args.cfg)

    cfg = parse_json(args.cfg)

    deep_learning_inference(cfg)
