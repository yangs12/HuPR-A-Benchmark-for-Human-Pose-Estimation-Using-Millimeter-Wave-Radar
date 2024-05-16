import numpy as np
from tqdm import tqdm

class Logger():
    def __init__(self):    
        self.bestAP = -1
        self.progressBar = None
        np.set_printoptions(precision=3)
    
    def clear(self, loaderSize):
        # return 
        self.progressBar = tqdm(total=loaderSize)
        
    def display(self, loss, loss2, updateSize, epoch):
        # return 
        if loss2 is not None:
            self.progressBar.set_postfix(EP=epoch, Loss=loss.item(), Loss2=loss2.item())
        else:
            self.progressBar.set_postfix(EP=epoch, Loss=loss.item())
        self.progressBar.update(updateSize)
    
    def showBestAP(self):
        # return 0
        return self.bestAP
    
    def isBestAccAP(self, acc):
        # return True
        if acc > self.bestAP or self.bestAP == -1:
            self.bestAP = acc
            return True
        else:
            return False