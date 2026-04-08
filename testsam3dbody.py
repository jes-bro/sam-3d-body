import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread('/home/jess/Downloads/cpr_vids/placetheotherhandontopofthefirst/nus_cpr_19_1/18.6457/cam01/output_001.png')
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
keypoints_3d_absolute_frame = outputs[0]["pred_keypoints_3d"] + outputs[0]["pred_cam_t"]  # figure out if this is right but this feels better? #[:, None, :] # understand this to make sure this is right 
keypoints_3d_relative_frame = outputs[0]["pred_keypoints_3d"] # I guess try doing it with both and creating visuals of both and see what happens 
keypoints_2d = outputs[0]["pred_keypoints_2d"]
# how do i account for differences in scale? 
# Figure out what the poses are really doing 
# Create different versions of retrieval and different images of each 
# Should I normalize the scale? But then that might fuck up the interactions no? 
# Hmm 
# Maybe the pose similarity will take care of egrigious differences in size + pose and the imagpose model can take care of it? 
# I raelly need to figure out the imagpose piece. 
# but once you have this you can generate 2d and 3d pose images and use both and the mesg pictures as well and train 3d models on each and see how they do and maybe try 
# the camera frame keypose and the normalized ones too run experiments with both? like comparing the normalized one and the non ? what does the normalization give you?
# is it really much different? where is the origin for the normalized one? 
print(keypoints_3d_absolute_frame) # these don't look like theyre in the pixel coord space. maybe it doesn't need to be if i fix and do the IMAGPose thing?
print(outputs[0].keys()) # these don't 
print(outputs[0]["pred_keypoints_2d"]) # do they just display these directly? if so, these are in the camera frame/pixel coord space, right? 
# read the paepr eat rn 
