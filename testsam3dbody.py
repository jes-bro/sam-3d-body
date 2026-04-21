import cv2
from PIL import Image
import os
import subprocess
import torch
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm
torch.cuda.empty_cache()




# need to do for each img in save folder yeah easier different folder per video easy to do low key cause already have the structure set up
# object_seg_2d = Image.open(output_path)
# obj_img_array = np.array(object_seg_2d)
# still you can do this all today or tomorrow 
# tomorrow 
 #havedone by friday 
 # and then where need to be today
 # get all images for all thingies today 
 # and run experiment tomorrow / figure out the training

# object_seg_2d.show()


# # Load and process image
# img_bgr = cv2.imread('/home/jess/Downloads/cpr_vids/placetheotherhandontopofthefirst/nus_cpr_19_1/18.6457/cam01/output_002.png')
## Iteratively create each 3D human body ## 
# imgs_path = '/home/jess/Downloads/cpr_vids/placetheotherhandontopofthefirst/nus_cpr_19_1/18.6457/cam01/'
video_paths = []
for root, dirs, files in os.walk("/home/jess/cpr_clips_5fps_copy/"):
    if "cam" in root.lower():
        video_paths.append(root)
        dirs.clear()  # stop recursing deeper into this branch
print(video_paths)
# exit()
# do for 

## Get all 2d obj mask images for given video at once ## 
# img_path = '/home/jess/Downloads/cpr_vids/placetheotherhandontopofthefirst/nus_cpr_19_1/18.6457/cam01/'
# make_2d_obj_mask_path = '/home/jess/sam3/make_mask.sh'
# output_path = os.path.join(img_path, "masks") # make correspond to input img figure that out
# # output_path = '/home/jess/sam-3d-body/test_img.png'
# subprocess.run(["bash", make_2d_obj_mask_path, img_path, output_path])
# video_paths = ...

# for path in video_paths: 
#     make_2d_obj_mask_path = '/home/jess/sam3/make_mask.sh'
#     output_path = os.path.join(path, "masks") # make correspond to input img figure that out
#     subprocess.run(["bash", make_2d_obj_mask_path, path, output_path])

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
video_count = len(video_paths)
count = 0
for path in video_paths: 
    percentage = (count / video_count) * 100
    print(f'doing video number {count} out of {video_count}')
    print(f'percentage done: {percentage}%')
    count += 1
    for root, dirs, file_names in os.walk(path):
        dirs[:] = []
        dirs.sort()
        for file in sorted(file_names):
            print('HERE IS THE FILE WE ARE CURRENTLY PROCESSING:', file)
            output_dir = os.path.join(root, "final_combined_masks")
            output_path = os.path.join(root, "final_combined_masks", file.replace(".png", "_mask.png"))
            if not os.path.exists(output_path):
                print('HERE IS THE OUTPUT DIRECTORY THAT WE ARE CREATING:', output_dir)
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(root, file)
                # print(f'file path/image name: {file_path}')
                obj_mask_path = os.path.join(root, "masks", file.replace(".png", "_mask.png"))
                if os.path.exists(obj_mask_path): 
                    print('HERE IS THE MASK WE ARE USING:', obj_mask_path)
                    img_bgr = cv2.imread(file_path)
                    print('HERE IS THE FINAL COMBINED HUMAN + OBJ MASK:', output_path)
                    os.makedirs(output_dir, exist_ok=True)
                    outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                        # print(f'corresponding mask: {obj_mask_path}')
                        # Visualize and save results
                    if outputs: 
                        rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
                        # if rend_img: 
                        body_3d_array = np.array(rend_img)
                        # wherever not black pixel, so all other pixels, set the values equal to those
                        human_3d_mask = np.any(body_3d_array, axis=-1)
                        object_mask_img = Image.open(obj_mask_path)
                        obj_img_array = np.array(object_mask_img)
                        obj_img_array[human_3d_mask] = body_3d_array[human_3d_mask]
                        final_combined_img = Image.fromarray(obj_img_array)
                        final_combined_img.save(output_path)
                # final_combined_img.show()
exit()
        
# cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
# print('SHAPES:')
# print(outputs[0]["pred_keypoints_3d"].shape)
# print(outputs[0]["pred_cam_t"].shape)
# keypoints_3d_absolute_frame = outputs[0]["pred_keypoints_3d"] + outputs[0]["pred_cam_t"]  # figure out if this is right but this feels better? #[:, None, :] # understand this to make sure this is right 
# keypoints_3d_relative_frame = outputs[0]["pred_keypoints_3d"] # I guess try doing it with both and creating visuals of both and see what happens 
# keypoints_2d = outputs[0]["pred_keypoints_2d"]
# # how do i account for differences in scale? 
# # Figure out what the poses are really doing 
# # Create different versions of retrieval and different images of each 
# # Should I normalize the scale? But then that might fuck up the interactions no? 
# # Hmm 
# # Maybe the pose similarity will take care of egrigious differences in size + pose and the imagpose model can take care of it? 
# # I raelly need to figure out the imagpose piece. 
# # but once you have this you can generate 2d and 3d pose images and use both and the mesg pictures as well and train 3d models on each and see how they do and maybe try 
# # the camera frame keypose and the normalized ones too run experiments with both? like comparing the normalized one and the non ? what does the normalization give you?
# # is it really much different? where is the origin for the normalized one? 
# # print(keypoints_3d_absolute_frame) # these don't look like theyre in the pixel coord space. maybe it doesn't need to be if i fix and do the IMAGPose thing?
# # print(outputs[0].keys()) # these don't 
# # print(outputs[0]["pred_keypoints_2d"]) # do they just display these directly? if so, these are in the camera frame/pixel coord space, right? 
# # read the paepr eat rn 

# a = torch.zeros((4, 3))
# print(a.shape)

# a[:, 2] = 3

# b = torch.zeros((3))
# b[:] = 3
# b[0] = 1
# b[1] = 2

# print(b.shape)
# print(a)
# print(b)
# print(a+b)
# print((a+b).shape)

