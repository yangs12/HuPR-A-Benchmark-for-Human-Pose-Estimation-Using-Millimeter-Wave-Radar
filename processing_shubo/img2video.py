import os
from moviepy.editor import *
import glob
from natsort import natsorted

clips = []
num = 190
num = str(num)
base_dir = "/home/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/data/HuPR/single_"+num+"/"

file_list = glob.glob(base_dir+'*.jpg')  # Get all the pngs in the current directory
file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
clips = [ImageClip(m).set_duration(1/10) for m in file_list_sorted]

# for filename in sorted(os.listdir(base_dir)):
    
#     if filename.endswith(".jpg"):
#         clips.append(ImageClip(base_dir+filename))
# print(clips)
video = concatenate(clips, method='compose')
print(video)
# video = concatenate_videoclips(clips, method="compose")

objectoutput_path = "/home/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/data/HuPR/"
name = "single_"+num+".mp4"
video.write_videofile(name, fps = 10.0)

