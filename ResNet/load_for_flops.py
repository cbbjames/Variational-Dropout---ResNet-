import pickle, pdb
import numpy as np

path ="experiments/weights/cifar10-vgglike-ard-1.0-zjw.npy"
path2 = "pretrain_resnet94_name.pickle"
with open(path, 'rb') as f:
  w_dict = np.load(f)
  
with open(path2, 'rb') as f:
  names = pickle.load(f)  
  
w_dich = []  

for i,w in enumerate(w_dict):
    if i%7==2: continue
    if len(w.shape)==4:
        ls2 = w_dict[i + 2] 
        log_alpha = ls2 - np.log(w ** 2)
        w[log_alpha > 3] = 0
        
        w = np.moveaxis(w, [0, 1, 2, 3], [3, 2, 0, 1]) #[3, 2, 0, 1] =>
    w_dich.append(w)    
var_dict = {n:w for n, w in zip(names,w_dich)}
pdb.set_trace()

with open("resnet_for_flops.pickle", 'wb') as f:
    pickle.dump(var_dict, f, protocol=3)
