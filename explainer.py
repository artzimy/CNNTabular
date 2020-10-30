from fastai.callbacks.hooks import *
from TabConvData import *
from TabConvModel import *

class Explainer:
    
    def __init__(self, data, learner, imsize=256):
        
        self.classes = data.classes
        self.data = data
        self.imsize = imsize
        self.learner = learner

    def init_explainer(self, idx):
        t = self.data.train_ds[idx][0].data
        x, y = self.data.train_ds[idx]
        xb,_ = self.data.one_item(x,denorm=True)
        xb = xb.cuda()
        self.xb = xb
        self.y = y
        self.m =  self.learner.model.eval();
        self.hook_a, self.hook_g = self.hooked_backward()
        
    def hooked_backward(self):
        with hook_output(self.m[0]) as hook_a: 
            with hook_output(self.m[0], grad=True) as hook_g:
                preds = self.m(self.xb)
                preds[0,int(self.y)].backward()

        return hook_a, hook_g
    
    def get_activations(self):
        acts  = self.hook_a.stored[0].cpu()
        avg_acts = acts.mean(0)
        self.activation =  avg_acts



    def create_heatmap(self):
        _,ax = plt.subplots()
        plt.title(f" True: {self.y}, Prediction: {self.data.classes[torch.argmax(self.m(self.xb))]} \npreds array: {[round(num, 3) for num in  self.m(self.xb).tolist()[0]]}", fontsize=16)
        xb_im = Image(self.xb[0])
        xb_im.show(ax)
        ax.imshow(self.activation, alpha=0.5, extent=(0,self.imsize, self.imsize,0),
                  interpolation='bilinear', cmap='jet')
       
    def show_heatmap(self, idx):
        self.init_explainer(idx)
        self.get_activations()
        self.create_heatmap()