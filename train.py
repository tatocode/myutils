import torch

class MyTrainer():
    def __init__(self, train_loader, val_loader, num_epochs, model, optim, loss_func, acc_func, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.model = model
        self.optim = optim
        self.loss_func = loss_func
        self.acc_func = acc_func
        self.device = device

        self.model = model.to(self.device)

    def train(self, save_weight=True):
        max_acc = 0
        for epoch in range(self.num_epochs):
            num = 0
            num_loss = 0
            self.model.train()
            for idx, data in enumerate(self.train_loader):
                img, label = data[0].to(self.device), data[1].to(self.device)
                out = self.model(img)
                loss = self.loss_func(out, label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                num_loss += loss.item()
                num += 1
            average_loss = num_loss/num
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

                        num_acc += acc
                        num += 1
                average_acc = num_acc/num
                print(f'{epoch+1}th epoch average acc: {average_acc:.4f}')

                if save_weight:
                    if epoch == 0:
                        torch.save(self.model.state_dict(), 'weights/best.pth')
                        torch.save(self.model.state_dict(), 'weights/last.pth')
                        max_acc = average_acc
                    else:
                        if average_acc > max_acc:
                            torch.save(self.model.state_dict(), 'weights/best.pth')
                        torch.save(self.model.state_dict(), 'weights/last.pth')

            elif save_weight:
                save = 20
                if epoch % save == 0:
                    torch.save(self.model.state_dict(), f'weights/epoch_{epoch+1}.pth')
