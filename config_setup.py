import importlib
import configlib
from config_utils import print_config
import sys
from types import SimpleNamespace
import ujson


project_list = [
    'Affine',
    'Longitudinal',
    'CrossMod',
    'ConditionalSeg',
    'Icn',
    'WeakSup',
    'CBCTUnetSeg',
    'mpmrireg',
    'GenSynthSeg'
]

def get_project():
    command_line = ' '.join(sys.argv)
    for pj in project_list:
        segment = f"--project {pj}"
        if segment in command_line:
            return pj
    raise NotImplementedError


#  some tips and tutorials about python import:
#  https://realpython.com/python-import/
#  print('project:', get_project() )
print(f"config.{get_project()}_train_config")
importlib.import_module(f"{get_project()}_train_config")

parser = configlib.add_parser("General config")
# General global options
parser.add_argument('--using_HPC', default=0, type=int, help='using UCL HPC')
parser.add_argument('--exp_name', default=None, type=str, help='experiment name you want to add.')
parser.add_argument('--project', default=None, type=str, help=f'the project name {project_list}')
parser.add_argument('--input_shape', default=[102, 102, 94], nargs='+', type=int, help='the shape of the images')
parser.add_argument('--voxel_size', default=[1.0, 1.0, 1.0], nargs='+', type=float, help='the size of the voxel')
parser.add_argument('--data_path', default=None, type=str, help='the path to the data')

# Dataloader options / augmentations
parser.add_argument('--affine_scale', default=0.0, type=float, help='affine transformation, scale 0 means not to add.')
parser.add_argument('--affine_seed', default=None, type=int, help='random seed for affine transformation')
parser.add_argument('--patched', default=0, type=float, help='take the cropped image patchs as network input')
parser.add_argument('--patch_size', default=[64, 64, 64], nargs='+', type=int, help='patch size, only used when --patched is 1.')
parser.add_argument('--inf_patch_stride_factors', default=[4, 4, 4], nargs='+', type=int, help='stride for getting patch in inference, stride=patchsize//this_factor')

# General Training options
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate.')
parser.add_argument('--batch_size', default=4, type=int, help='The number of batch size.')
parser.add_argument('--gpu', default=0, type=int, help='id of gpu')
parser.add_argument('--num_epochs', default=3000, type=int, help='The number of iterations.')
parser.add_argument('--save_frequency', default=10, type=int, help='save frequency')
parser.add_argument('--continue_epoch', default='-1', type=str, help='continue training from a certain ckpt')

config = SimpleNamespace(**configlib.parse())

assert config.exp_name is not None, "experiment name should be set"
assert config.data_path is not None, "data path is not provided"

print_config(config, parser)

# with open('./gen_config.json', 'w') as f:
#     ujson.dump(vars(config), f)