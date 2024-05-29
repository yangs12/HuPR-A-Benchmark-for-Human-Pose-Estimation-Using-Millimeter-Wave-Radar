import h5py
import numpy as np

folder = 178
file_name = 1
file_name_9 = str(file_name).zfill(9)
path = f'/home/jupyter/hupr/net_preprocessed/single_{folder}/{file_name_9}.npy'

file = np.load(path, allow_pickle=True)
hori = file.item().get('hori')
vert = file.item().get('vert')
# print(hori.shape)
# print(hori.dtype)

hori_net = np.load('hupr_data_test/VRDAmaps_hori_single15_599.npy').astype(np.float16)
# np.load('hupr_data_test/VRDAEmaps_hori_'+str(folder)+'_'+str(file_name)+'.npy')
# hori_net = np.mean(hori_net, axis=-1).astype(np.float16)
print(hori_net.shape)
vert_net = np.load('hupr_data_test/VRDAmaps_vert_single15_599.npy').astype(np.float16)
# vert_net = np.load('hupr_data_test/VRDAEmaps_vert_'+str(folder)+'_'+str(file_name)+'.npy')
# vert_net = np.mean(vert_net, axis=-1).astype(np.float16)

# print(np.load('VRDAmaps_hori_single3_599.npy').dtype)
print('Hori', (hori == hori_net).all())
print('Vert', (vert == vert_net).all())
# print('horinet', hori_net[0,0,0,:])
# print('hori', hori[0,0,0,:])


# import numpy as np
# VRDAERealImag_hori = np.load('/home/jupyter/hupr/radar_processed/single_1/hori/000000000.npy', allow_pickle=True)
# print(VRDAERealImag_hori.shape)
# print(VRDAERealImag_hori.dtype)


# read hdf5

# path = '/home/jupyter/hupr/net_preprocessed/single_1/000000010.hdf5'
# file = h5py.File(path, 'r')
# hori = np.array(file.get('hori'))
# vert = np.array(file.get('vert'))
# print(vert.shape)