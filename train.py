import json, datetime, os
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from data import CSVDataManager
from net import AttentionalFactorizationMachine, FactorizationMachine
from utils import WriterTensorboardX, mkdir_p

class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        

        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.log_path = os.path.join(config['train']['save_dir'], start_time)

        tb_path = os.path.join(self.log_path, 'logs')
        mkdir_p(tb_path)
        self.writer = WriterTensorboardX(tb_path)
        
        data_manager = CSVDataManager(config['data'])
        self.data_loader = data_manager.get_loader('train')
        self.valid_data_loader = data_manager.get_loader('val')
        
        self.model = AttentionalFactorizationMachine(data_manager.dims, config)
        self.model = self.model.to(self.device)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = SGD(trainable_params, **config['optimizer'])
        self.lr_scheduler = StepLR(self.optimizer, **config['lr_scheduler'])

        self.best_val_loss = float('inf')
        self.satur_count = 0

    def _train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0
        self.writer.set_step(epoch)
        _trange = tqdm(self.data_loader, leave=True, desc='')

        for batch_idx, batch in enumerate(_trange):
            batch = [b.to(self.device) for b in batch]

            data, target = batch[:-1], batch[-1]
            # data -> users, items, gens
        
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:                
                _str = 'Train Epoch: {} Loss: {:.6f}'.format(epoch,loss.item()) 
                _trange.set_description(_str)

        loss = total_loss / len(self.data_loader)
        self.writer.add_scalar('loss', loss)

        log = {'loss': loss}

        val_log = self._valid_epoch(epoch)
        log = {**log, **val_log}

        self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):

        self.model.eval()
        total_val_loss = 0

        self.writer.set_step(epoch, 'valid')        

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = [b.to(self.device) for b in batch]

                data, target = batch[:-1], batch[-1]
            
                output = self.model(data)
                loss = F.mse_loss(output, target)

                total_val_loss += loss.item()

            val_loss = total_val_loss / len(self.valid_data_loader)
            
            self.writer.add_scalar('loss', val_loss)
                    
            #for name, param in self.model.named_parameters():
            #    if param.requires_grad:
            #        self.writer.add_histogram(name, param.clone().cpu().numpy(), bins='doane')

        return {'val_loss': val_loss}


    def train(self):
        print(self.model)

        for epoch in range(1, self.config['train']['epochs'] + 1):

            result = self._train_epoch(epoch)

            c_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('lr', c_lr)

            log = pd.DataFrame([result]).T
            log.columns = ['']  
            print(log)

            if self.best_val_loss > result['val_loss']:
                print('[IMPROVED]')
                chk_path = os.path.join(self.log_path, 'checkpoints')
                mkdir_p(chk_path)

                state = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }

                torch.save(state, os.path.join(chk_path, 'model_best.pth'))
                with open(os.path.join(chk_path, 'config.json'), 'w') as wj:
                    json.dump(self.config, wj)
            else:
                self.satur_count += 1

            if self.satur_count > self.config['train']['early_stop']:
                break

if __name__ == '__main__':

    import json

    config = json.load(open('config.json', 'r'))
    m = Trainer(config)
    m.train()