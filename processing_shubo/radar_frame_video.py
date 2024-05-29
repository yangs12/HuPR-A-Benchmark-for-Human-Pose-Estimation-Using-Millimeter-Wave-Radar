import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from moviepy.editor import *
import glob
from natsort import natsorted

folder_num = 276
folder = 'vert'
save_images = True
RD = False

if save_images:
    # -------RD--------        
    if RD:
        output_dir = '/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/figs_RD/'+folder+'/'
        os.makedirs(output_dir, exist_ok=True)
     
        for i in range(600):
            data = np.load('/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/'+folder+'/0000' + f'{i:05d}' + '.npy')
            # print('Preprocessed data shape: ', data.shape) # (16, 64, 64, 8) Doppler, Range, Azimuth, Elevation
            RD = np.average(data, axis=3)
            RD = np.average(RD, axis=2) 

            plt.imshow(np.abs(RD))
            # plt.colorbar()
            plt.xlabel('Range (m)')
            plt.ylabel('Velocity (m/s)')
            range_reso = 4.3/100
            x = np.round(np.linspace(0, RD.shape[1], 10),2)
            plt.xticks(x, np.round(30*range_reso+x*range_reso,2))
            y = np.round(np.linspace(24, 40, 6),2)
            plt.yticks(np.round(np.linspace(0, 15, 6),2), np.round((y-32)*0.42,2))
            
            plt.savefig('/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/figs_RD/'+folder+'/0000' + f'{i:05d}' + '.png')
            plt.close()

    # -------AE--------
    else:
        output_dir = '/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/figs_AE/'+folder+'/'
        os.makedirs(output_dir, exist_ok=True)
        for i in range(600):
            data = np.load('/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/'+folder+'/0000' + f'{i:05d}' + '.npy')
            AE = np.average(data, axis=0)
            AE = np.average(AE, axis=0) 

            if folder == 'vert':
                AE = np.transpose(AE)
            plt.imshow(np.transpose(np.abs(AE)))

            if folder == 'hori':
                plt.ylabel('Elevation ($^\circ$)')
                plt.xlabel('Azimuth ($^\circ$)')
                x = np.round(np.linspace(0, AE.shape[0]-1, 9),2)
                xtick = np.round(np.linspace(-60, 60, 9),2)
                plt.xticks(x, xtick)
                y = np.round(np.linspace(0, AE.shape[1]-1, 3),2)
                ytick = np.round(np.linspace(-15, 15, 3),2)
                plt.yticks(y, ytick)
            else:
                plt.ylabel('Azimuth ($^\circ$)')
                plt.xlabel('Elevation ($^\circ$)')
                x = np.round(np.linspace(0, AE.shape[0]-1, 3),2)
                xtick = np.round(np.linspace(-15, 15, 3),2)
                plt.xticks(x, xtick)
                y = np.round(np.linspace(0, AE.shape[1]-1, 9),2)
                ytick = np.round(np.linspace(-60, 60, 9),2)
                plt.yticks(y, ytick)
            plt.savefig('/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/figs_AE/'+folder+'/0000' + f'{i:05d}' + '.png')
            plt.close()


clips = []
file_list = glob.glob(output_dir+'*.png')  # Get all the pngs in the current directory
file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
clips = [ImageClip(m).set_duration(1/10) for m in file_list_sorted]

video = concatenate(clips, method='compose')

if RD:
    name = "radar_single_"+str(folder_num)+'_'+folder+"_RD.mp4"
else:
    name = "radar_single_"+str(folder_num)+'_'+folder+"_AE.mp4"
video.write_videofile('/home/jupyter/hupr/radar_processed/single_'+str(folder_num)+'/' + name, fps = 10.0)
