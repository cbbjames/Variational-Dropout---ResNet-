import sys, os, pdb
import numpy as np
import pickle

path = "pretrain_resnet94_new.pickle"
with open(path, 'rb') as f:
  w_dict = pickle.load(f, encoding='latin1')
  f.close()
  
path = "pretrain_resnet94_dict_new.pickle"
with open(path, 'rb') as f:
  name_arr = pickle.load(f, encoding='latin1')  
  f.close()

name_arr = name_arr[2:189]
w_arr = w_dict[2:189]




w_id = 0  
name_id = 0

#conv 0 
bias = np.zeros((64,))
w_arr = w_arr[:7]+[bias]+w_arr[7:]
name_arr = name_arr[:7]+['bias']+name_arr[7:]
'''
for n, w in zip(name_arr, w_arr):
  print(n, w.shape)
pdb.set_trace()   
'''
#first_block = [64, 3, h, w]

#{
#first_block = [out, in, h, w]
#second block = [out, in, h, w]

#block = [c, c, h, w]
# *8
#                               }*3
# fc

# w, b, beta, gamma, => W, b, ls2, beta, gamma, mean, var    
'''
for v in w_arr:
  print(v.shape)
'''
params = []
names = []
init = 15
for i in range(0, len(name_arr)-2, 6):

  w = w_arr[i]
  w = np.moveaxis(w, [0, 1, 2, 3], [-2, -1, 1, 0]) #[3, 2, 0, 1 ]
  
  b = w_arr[i+1]
    
  mean = w_arr[i+2]
  var = w_arr[i+3]
  
  beta = w_arr[i+4]
  gamma = w_arr[i+5]

  ls2 = np.zeros_like(w)-init
  
  names+=['W', 'b', 'ls2', 'beta', 'gamma', 'mean', 'var']
  params+=[w, b, ls2, beta, gamma, mean, var]

params+=w_arr[-2:]
names+=name_arr[-2:]

params.append(np.zeros_like(w_arr[-2])-init)
names.append('ls2')

assert len(params)==len(names)
for s, name in zip(params, names):
  print(name, s.shape)
pdb.set_trace()
np.save("pretrain_res945.npy", params, fix_imports=True)