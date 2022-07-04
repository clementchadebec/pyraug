import configlib



parser = configlib.add_parser("CT and CBCT segmentation configs")

# Network options
parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')
parser.add_argument('--inc', default=2, type=int, help='input channel number of the network, if 3, mv_seg will be feed into the network')
parser.add_argument('--outc', default=2, type=int, help='output channel number (label class num), consider if count in the backgroun')
# Training options
parser.add_argument('--model', default='UNet', type=str, help='LocalAffine/LocalEncoder/LocalModel/...')
parser.add_argument('--cv', default=0, type=int, help='The fold for cross validation')
#parser.add_argument('--continue_epoch', default=-1, type=int, help='Selst to -1 to disallow training from most recent epoch. Set to epoch # to continue training.')

# sampling options
parser.add_argument('--input_mode', default='both', type=str, help='ct/cbct/oneof/both')
parser.add_argument('--two_stage_sampling', default=1, type=int, help='Only in training, random pick up one label for each sample.')
parser.add_argument('--crop_on_seg_aug', default=0, type=int, help='adding random crop to the segmentaion')
parser.add_argument('--patient_cohort', default='intra', type=str, help='Only in training, input inter or intra pairs')


# loss & weights
parser.add_argument('--w_dce', default=1.0, type=float, help='the weight of dice loss')
parser.add_argument('--w_bce', default=0, type=float, help='the weight of weighted binary cross-entropy')
parser.add_argument('--dice_class_weights', default=[1, 2], nargs='+', type=float, help='weights for each class in dice loss')


