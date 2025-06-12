import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
# import moviepy.video.io.ImageSequenceClip
import moviepy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import VideoFileClip 
# from moviepy.editor import *
import torch
import matplotlib.pyplot as plt
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
from gelslim_shear.plot_utils.shear_plotter import plot_vector_field, plot_scalar_field, get_channel
import torch.nn.functional as F
from tqdm import tqdm
import time

def make_video(images, video_name, fps=30):
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(video_name, logger=None)
    return video_name

def square_center_crop(image):
    height = image.shape[1]
    width = image.shape[2]
    if height > width:
        start = (height - width)//2
        return image[:, start:start+width, :]
    elif width > height:
        start = (width - height)//2
        return image[:, :, start:start+height]
    else:
        return image
    
def downsample(image, size):
    return F.interpolate(image.unsqueeze(0), size=size, mode='area').squeeze(0)



# Folder paths
input_folder = './Processing Queue'
output_folder = './Processed Videos'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# try cuda and print if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)

frame_period = 1 / 30  # Adjust frame period as needed (FPS is assumed to be 30)

shgen = ShearGenerator(method='2', channels=['u', 'v', 'div', 'du', 'dv'], 
                       Farneback_params=(0.5, 3, 45, 3, 5, 1.2, 0), output_size=(18, 18))

# Iterate through all video files in the "Processing Queue"
for video_path in Path(input_folder).glob("*.mp4"):  # assuming all videos are in .mp4 format
    original_video = str(video_path)
    video_name = f"Shear_{video_path.stem}.mp4"  # Add 'Shear_' prefix to the output video name
    temp_folder = 'temp'

    clip = VideoFileClip(original_video)
    duration = clip.duration
    fps = clip.fps
    frame_period = 1 / fps 

    # Create a temporary folder for images
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for frame in tqdm(range(int(duration * fps))):
        # MoviePy 1.0.3: get_frame() takes time (seconds), not frame index

        time0 = time.time()

        frame_time = frame * frame_period
        frame_image = torch.from_numpy(clip.get_frame(frame_time)  ).permute(2, 0, 1).float().to(device)
        frame_image = square_center_crop(frame_image)
        frame_image = downsample(frame_image, (200, 200))

        if frame == 0:
            shgen.update_base_tactile_image(frame_image)
        shgen.update_time(frame * frame_period)
        shgen.update_tactile_image(frame_image)
        shgen.update_shear()
        shear_field_tensor = shgen.get_shear_field()


        vf = get_channel(shear_field_tensor, [shgen.channels.index('u'), shgen.channels.index('v')])
        sf = get_channel(shear_field_tensor, shgen.channels.index('div'))
        diff_vf = get_channel(shear_field_tensor, [shgen.channels.index('du'), shgen.channels.index('dv')])
        print(vf.shape,sf.shape,diff_vf.shape)

        # ax[0, 0].imshow(frame_image.permute(1, 2, 0).cpu().numpy().astype('uint8'))
        # ax[0, 0].axis('off')
        # ax[0, 0].set_xticks([])
        # ax[0, 0].set_yticks([])
        # plot_vector_field(ax[0, 1], vf, title='Shear Field', color='blue')
        # plot_scalar_field(ax[1, 0], sf, title='Divergence', max_magnitude=3)
        # plot_vector_field(ax[1, 1], diff_vf, title='Change in Shear Field', color='red')
        # fig.tight_layout()
        # fig.savefig(f'{temp_folder}/{frame}.png')
        # [ax[i, j].cla() for i in range(2) for j in range(2)]

    # Create video from processed frames
    # images = [f'{temp_folder}/{frame}.png' for frame in range(int(duration * fps))]
    # make_video(images, os.path.join(output_folder, video_name), fps=fps)

    # Clean up temporary images
    # for image in images:
    #     os.remove(image)
