from utils import logging
import os

import torchvision.utils

from utils import draw
import torch

class MyTrainer():
    def __init__(self, train_config_dict):
        self.train_loader = train_config_dict['train_loader']
        self.val_loader = train_config_dict['val_loader']
        self.num_epochs = train_config_dict['num_epochs']
        self.optim = train_config_dict['optim']
        self.loss_func = train_config_dict['loss_func']
        self.acc_func = train_config_dict['acc_func']
        self.device = train_config_dict['device']
        self.model = train_config_dict['model'].to(self.device)

    def train(self, home_dir, save_weight=True, print_log=True, show_val=True):
        print(f'use device: {self.device}')
        max_acc = 0
        if print_log:
            logger = logging.getLogger(home_dir)
        train_loss = []
        val_acc = []
        if not os.path.exists(home_dir):
            os.mkdir(home_dir)
        if not os.path.exists(os.path.join(home_dir, 'weights')) :
            os.mkdir(os.path.join(home_dir, 'weights'))
        for epoch in range(self.num_epochs):
            num = 0
            num_loss = 0
            self.model.train()
            for idx, data in enumerate(self.train_loader):
                img, label = data[0].to(self.device), data[1].to(self.device)
                out = self.model(img)
                # print(out)
                loss = self.loss_func(out, label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                num_loss += loss.item()
                num += 1
            average_loss = num_loss/num
            train_loss.append(average_loss)
            if print_log:
                logger.info(f'train|epoch:{epoch+1}\t loss:{average_loss:.4f}')
            else:
                print(f'{epoch+1}th epoch average loss: {average_loss:.4f}')

            if self.val_loader is not None:
                self.model.eval()
                num = 0
                num_acc = 0
                with torch.no_grad():
                    for idx, data in enumerate(self.val_loader):
                        img, label = data[0].to(self.device), data[1].to(self.device)
                        out = self.model(img)
                        acc = self.acc_func(out, label)

                        if show_val:
                            # 画图
                            if os.path.exists(home_dir+'/val_imgs'):
                                os.mkdir(home_dir+'/val_imgs')
                            draw_img = img[0,:, :, :].cpu()
                            draw_label = label[0,:,:].cpu()*(torch.ones(3*224*224).resize(3, 224, 224)*255)
                            draw_pre = out[0,:,:].cpu()*(torch.ones(3*224*224).resize(3, 224, 224)*255)
                            torchvision.utils.save_image(torch.stack([draw_img, draw_label, draw_pre], dim=0), home_dir+'/val_imgs'+f'/show{epoch+1}.png')

                        num_acc += acc
                        num += 1
                average_acc = num_acc/num
                val_acc.append(average_acc.cpu())
                if print_log:
                    logger.info(f'val|epoch:{epoch+1}\t acc:{average_acc:.4f}')
                else:
                    print(f'{epoch+1}th epoch average acc: {average_acc:.4f}')

                if save_weight:
                    if epoch == 0:
                        torch.save(self.model.state_dict(), os.path.join(home_dir, 'weights/best.pth'))
                        torch.save(self.model.state_dict(), os.path.join(home_dir, 'weights/last.pth'))
                        max_acc = average_acc
                    else:
                        if average_acc > max_acc:
                            torch.save(self.model.state_dict(), os.path.join(home_dir, 'weights/best.pth'))
                        torch.save(self.model.state_dict(), os.path.join(home_dir, 'weights/last.pth'))

            elif save_weight:
                save = 20
                if epoch % save == 0:
                    torch.save(self.model.state_dict(), os.path.join(home_dir, f'weights/epoch_{epoch+1}.pth'))
        draw.draw_train_val_graph(train_loss, val_acc, home_dir)