import torch
import os
import numpy as np
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
# from huggingface_hub import login
import cv2
from pathlib import Path
import gc
import sys
# gc.collect()
# torch.cuda.empty_cache()

# login()
# Load the model
model = build_sam3_image_model(device='cuda')
# video_predictor = build_sam3_video_predictor()


# model.to(device="cpu")

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def make_masks(jpg_path, mask_path):
    print(f"video path: {jpg_path}")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=jpg_path,
        )
    )
    id = response["session_id"]
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=response["session_id"],
            frame_index=0, # Arbitrary frame index
            text="cpr mannequin", # one human providing cpr - pretty good human prompt. maybe should have been more vague? could try again but think i should move on for now to get to fine-tuning and the stick figures 
        )
    )

    human_output = propagate_in_video(video_predictor, id)
    # human_masks = human_output["out_binary_masks"]
    for mask in human_output.keys():
        print(human_output[mask]["out_binary_masks"].shape)
        human_mask = human_output[mask]["out_binary_masks"] # (1, 448,796)
        if human_mask.shape[0] != 0:
            areas = human_mask.reshape(human_mask.shape[0], -1).sum(axis=1)
            human_mask = human_mask[np.argmax(areas)]         
        # mannequin_mask = obj_mask.detach().cpu().numpy()
        human_np = human_mask.squeeze() 
        human_np = (human_np > 0).astype(np.uint8) * 255
        rgb = np.zeros((human_np.shape[0], human_np.shape[1], 3), dtype=np.uint8)
        mask_loc = os.path.join(mask_path, f'mask_{mask}.png')
        outdir = Path(mask_path)
        outdir.mkdir(parents=True, exist_ok=True)
        if human_np.shape[0] > 0:
            rgb[...,0] = human_np
            img = Image.fromarray(rgb, mode="RGB")
            img.save(os.path.join(mask_path, f'mask_{mask}.png'))
            print(f"saved mask to {mask_loc}")

        else:
            print(f"mask {mask_loc} was empty")
    del response
    
    # obj_np = mannequin_mask.squeeze() 
    # obj_np = (obj_np > 0).astype(np.uint8) * 255
    # print(mask_np.shape, mask_np.dtype, mask_np.min(), mask_np.max())
    # human_mask_img = Image.fromarray(human_np, mode="RGB")
    # obj_mask_img = Image.fromarray(obj_np, mode="RGB")
    # combined_rgb = np.zeros((human_np.shape[0], human_np.shape[1], 3), dtype=np.uint8)
    # # obj_rgb = np.zeros((human_np.shape[0], human_np.shape[1], 3))
    # outdir = Path(mask_path)
    # outdir.mkdir(parents=True, exist_ok=True)
    # mask_name = img_name.split('_')[1].replace('jpg', 'png')
    # mask_name = 'mask_' + mask_name
    # if human_np.shape[0] > 0:
    #     combined_rgb[...,1] = human_np
    # if obj_np.shape[0] > 0:
    #     combined_rgb[...,0] = obj_np
    # combined_img = Image.fromarray(combined_rgb, mode="RGB")
    # # out_dir_name = dir + '/masks'
    
    # combined_img.save(os.path.join(mask_path, mask_name))
    # print(human_output.keys())
    # gc.collect()
    # torch.cuda.empty_cache()

    # response = video_predictor.handle_request(
    #     request=dict(
    #         type="start_session",
    #         resource_path=jpg_path,
    #     )
    # )
    # response = video_predictor.handle_request(
    #     request=dict(
    #         type="add_prompt",
    #         session_id=response["session_id"],
    #         frame_index=0, # Arbitrary frame index
    #         text="mannequin lying down",
    #     )
    # )
    # obj_output = response["outputs"]
    # obj_masks = obj_output["out_binary_masks"]
    # print(obj_masks.shape)
    # del response
    # print(obj_output.keys())

def make_mask(img_file_path, output_path):
    
    processor = Sam3Processor(model)
    processor2 = Sam3Processor(model)
    # Load an image
    print(f"path to open: {img_file_path}")
    image = Image.open(img_file_path)
    inference_state = processor.set_image(image)
    inference_state2 = processor2.set_image(image)
    # Prompt the model with text
    human_output = processor.set_text_prompt(state=inference_state, prompt="human")
    obj_output = processor.set_text_prompt(state=inference_state2, prompt="cpr mannequin")
    print("Got here")
    human_mask = human_output["masks"]
    areas = human_mask.flatten(1).float().mean(dim=1)     # (2,)
    if human_mask.numel() > 0 and human_mask.shape[0] > 0:
        human_mask = human_mask[areas.argmax().item()]           # (448,796)
    obj_mask = obj_output["masks"]
    obj_areas = obj_mask.flatten(1).float().mean(dim=1)     # (2,)
    if obj_mask.numel() > 0 and obj_mask.shape[0] > 0:
        obj_mask = obj_mask[obj_areas.argmax().item()]           # (448,796)

    # if human_mask.shape[0] > 1:
    #     human_mask = human_mask[0,...]
    print("type:", type(human_mask))
    if isinstance(human_mask, torch.Tensor):
        print("shape:", tuple(human_mask.shape))
    # # cv2.imwrite("mask_test.png", mask)
    # print(human_output["masks"].shape)
    # print(obj_output["masks"].shape)
        print("numel:", human_mask.numel())
        print("dtype:", human_mask.dtype)
        print("device:", human_mask.device)
    # print(max(human_output["masks"]))
    # print(max(obj_output["masks"]))
    # mask = human_output["masks"] | obj_output["masks"]
    print("type:", type(obj_mask))
    if isinstance(obj_mask, torch.Tensor):
        print("shape:", tuple(obj_mask.shape))
        print("numel:", obj_mask.numel())
        print("dtype:", obj_mask.dtype)
        print("device:", obj_mask.device)
    print("Here too")
    human_mask = human_mask.detach().cpu().numpy()
    mannequin_mask = obj_mask.detach().cpu().numpy()
    human_np = human_mask.squeeze() 
    human_np = (human_np > 0).astype(np.uint8) * 255
    obj_np = mannequin_mask.squeeze() 
    obj_np = (obj_np > 0).astype(np.uint8) * 255
    # print(mask_np.shape, mask_np.dtype, mask_np.min(), mask_np.max())
    # human_mask_img = Image.fromarray(human_np, mode="RGB")
    # obj_mask_img = Image.fromarray(obj_np, mode="RGB")
    combined_rgb = np.zeros((obj_mask.shape[1], obj_mask.shape[2], 3), dtype=np.uint8)
    # obj_rgb = np.zeros((human_np.shape[0], human_np.shape[1], 3))
    # outdir = Path(mask_path)
    # outdir.mkdir(parents=True, exist_ok=True)
    # mask_name = img_name.split('_')[1].replace('jpg', 'png')
    # mask_name = 'mask_' + mask_name
    if human_np.size > 0 and human_np.shape[0] > 0:
        print("Adding the human")
        # combined_rgb[...,1] = human_np # [...,1]
        print("Human mask created")
        # pass
    if obj_np.size > 0 and obj_np.shape[0] > 0:
        combined_rgb[...,0] = obj_np
        print("Adding the object")
    combined_img = Image.fromarray(combined_rgb, mode="RGB")
    # out_dir_name = dir + '/masks'
    # output_path = os.path.join(mask_path, mask_name)
    # np.save(output_path, combined_rgb)
    print(f"img saved to {output_path}")
    combined_img.save(output_path)


    # print(type(mask))
    # # cv2.imwrite("mask_test.png", mask)
    # print(human_output["masks"].shape)
    # print(obj_output["masks"].shape)

    # Get the masks, bounding boxes, and scores
    # results = output["masks"], output["boxes"], output["scores"]

    # plot_results(image, outpute)e
# from PIL import Image 
# dirs = "/home/jess/Downloads/new"
# for dir in os.listdir(dirs):
#     # print(dir)
#     # img_path = os.path.join(dirs, dir, 'images0', 'frames_seq')
#     mask_path = os.path.join(dirs, dir, 'masks')
#     mask_obj_path = os.path.join(dirs, dir, 'masks_obj')
#     out_path = os.path.join(dirs, dir, 'combined_masks')
#     # print(img_path)
#     for img in os.listdir(mask_path):
#         full_human_mask_path = os.path.join(mask_path, img)
#         full_obj_mask_path = os.path.join(mask_obj_path, img)
#         print(full_human_mask_path)
#         Image.open(full_obj_mask_path).verify()
#         Image.open(full_human_mask_path).verify()
#         print("PIL says OK")
#         print(full_obj_mask_path)
#         outdir = Path(mask_path)
#         outdir.mkdir(parents=True, exist_ok=True)
#         r_mask = np.array(Image.open(full_obj_mask_path))
#         g_mask = np.array(Image.open(full_human_mask_path))
#         out = np.zeros_like(r_mask)
#         out[..., 0] = r_mask[..., 0]  # R channel
#         out[..., 1] = g_mask[..., 1]  # G channel

#         Image.fromarray(out).save(os.path.join(outdir, img))
    #     # print(mask_path)
    #     if not img == "frames_seq":
    #         img_file_path = os.path.join(img_path, img)
    #         # print(img_file_path)
    #         make_mask(img_file_path, mask_path, img)
    # make_masks(img_path, mask_path
input_path = sys.argv[1]
output_path = sys.argv[2]
print(input_path)
print(output_path)
# input_path = "/home/jess/Downloads/cpr_vids/presshardatarateof100to120compressionsperminute/nus_cpr_11_1/27.676/cam01/output_001.png"
make_mask(input_path, output_path)
# make_mask("/home/jess/Downloads/cpr_pose_test/expert_pose_cpr.jpg", "/home/jess/sm", "_expertmannequinmask.jpg")
