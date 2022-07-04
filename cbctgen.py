import dataloaders
from torch.utils.data import DataLoader


class CBCTGenerator():
    def __init__(self, config):
        #super(cbctSeg, self).__init__(config)
        self.config = config
        #self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0


    def set_dataloader(self):
        self.train_set = dataloaders.CBCTData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size,  
            num_workers=4, 
            shuffle=True,
            drop_last=True)
        print('>>> Train set ready.')  
        self.val_set = dataloaders.CBCTData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.CBCTData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready.')


    def get_input(self, input_dict, aug=True):
        fx_img, mv_img = input_dict['fx_img'].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z], image
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()  # label
        fx_seg, mv_seg = fx_seg[:, 0, ...], mv_seg[:, 0, ...]

        if (self.config.affine_scale != 0.0) and aug:
            mv_affine_grid = smfunctions.rand_affine_grid(
                mv_img, 
                scale=self.config.affine_scale, 
                random_seed=self.config.affine_seed
                )
            fx_affine_grid = smfunctions.rand_affine_grid(
                fx_img, 
                scale=self.config.affine_scale,
                random_seed=self.config.affine_seed
                )
            mv_img = torch.nn.functional.grid_sample(mv_img, mv_affine_grid, mode='bilinear', align_corners=True)
            mv_seg = torch.nn.functional.grid_sample(mv_seg, mv_affine_grid, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, fx_affine_grid, mode='bilinear', align_corners=True)
            fx_seg = torch.nn.functional.grid_sample(fx_seg, fx_affine_grid, mode='bilinear', align_corners=True)
        else:
            pass

        # ct/cbct/oneof/both
        if self.config.input_mode == 'both':
            assert self.config.inc == 2, "input channel needs to be 2"
            return torch.cat([fx_img, mv_img], dim=1), fx_seg  # [cbct, ct], cbct_seg
        elif self.config.input_mode == 'ct':
            assert self.config.inc == 1, "input channel needs to be 1"
            if self.phase == 'train':
                return mv_img, mv_seg
            else:
                return fx_img, fx_seg
        elif self.config.input_mode == 'cbct':
            assert self.config.inc == 1, "input channel needs to be 1"
            return fx_img, fx_seg
        elif self.config.input_mode == 'oneof':
            assert self.config.inc == 1, "input channel needs to be 1"
            if self.phase == 'train':
                tmp = [(fx_img, fx_seg), (mv_img, mv_seg)]
                return random.sample(tmp, 1)[0]
            else:
                return fx_img, fx_seg
        else:
            raise NotImplementedError