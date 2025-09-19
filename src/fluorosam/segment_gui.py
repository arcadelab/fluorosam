import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from pathlib import Path
from pydicom import dcmread
from .model_wrapper import SALitModule
from .build_model import create_sam_model
from .utils import image_utils
import logging
import argparse

# Configure Logging
log = logging.getLogger("segment")

RED = [145, 56, 49]
MAGENTA = [197, 58, 224]
ckpt_path = None

def pad_to_square(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pads the image to make it square."""
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape[:2]

    if h == w:
        return image, (0, 0, 0, 0)

    val = np.mean(image)
    if h > w:
        pad = (h - w) // 2
        if len(image.shape) == 3:
            return np.pad(image, ((0, 0), (pad, pad), (0, 0)), mode="constant", constant_values=val), (0, 0, pad, pad)
        else:
            return np.pad(image, ((0, 0), (pad, pad)), mode="constant", constant_values=val), (0, 0, pad, pad)
    else:
        pad = (w - h) // 2
        if len(image.shape) == 3:
            return np.pad(image, ((pad, pad), (0, 0), (0, 0)), mode="constant", constant_values=val), (pad, pad, 0, 0)
        else:
            return np.pad(image, ((pad, pad), (0, 0)), mode="constant", constant_values=val), (pad, pad, 0, 0)


def segment(prompt: list[str], input_path: Path, output_dir: Path | None, size: int | None, multimasks: bool = False, points: list[tuple] = None) -> str:
    """Runs segmentation with the given image and prompt. Returns the path to the output image."""
    image_path = Path(input_path)
    og_image = image_label.og_image
    img_shape = og_image.shape[:2]
    print(img_shape)

    global ckpt_path

    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        return None

    model = create_sam_model(backbone="swin-l", pretrained=True, weight_path=str(ckpt_path))
    model = SALitModule(model)
    model.eval()

    image_dir = output_dir / image_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for i, p in enumerate(prompt):
        log.info(f"\nPredicting mask for the prompt '{p}'")
        image_embedding = model.encode_image(og_image, mixture=True)  # Ensure embedding is created
        if points is not None:
            point = points[i]
            new_point = (np.array([[point]]), np.array([[1]]))
            point = (point[0]/img_shape[0] * 448, point[1]/img_shape[1] * 448)
            # format the point so its a tuple of arrays where the firs array is (n,2) 
            # with the point and the second is (n,1) tha indicates the point belongs to the object
            input_point = (np.array([[point]]), np.array([[1]]))
            input_points = image_label.input_points
            input_points.extend(input_point[0][0])
            image_label.input_points = input_points
            n_points = len(input_points)
            point = (np.array([input_points]), np.ones((1, n_points)))
        else:
            point = None
    
        mask, pred_iou, pred_multimasks = model.predict_mask(
            image_embedding, p, point, original_size=og_image.shape[:2], return_iou=True
        )
        if multimasks:
            masks = list(pred_multimasks)
            break
        else:
            mask = mask.reshape(1, *mask.shape)
            masks.extend(mask)

    points_in_image = image_label.points_image
    if points is not None:
        # resize the point to the original size
        point_vis = new_point[0][0][0]
        # point_vis = (int(point_vis[0] * og_image.shape[1] / 400), int(point_vis[1] * og_image.shape[0] / 400))
        point_vis = np.array([point_vis])
        points_in_image.extend(point_vis)
        point_vis = np.array(points_in_image)
        mask_vis = image_utils.draw_masks(og_image, masks, threshold=0.5, contour_thickness=3)
        mask_vis = image_utils.draw_keypoints(mask_vis, point_vis)
        image_label.points_image = points_in_image
    else:
        mask_vis = image_utils.draw_masks(og_image, masks, threshold=0.5, contour_thickness=3)

    latest_output = image_dir / f"{i:02d} {p} ({pred_iou:.02f}).png"
    Image.fromarray(mask_vis).save(latest_output)
    log.info(f"Saved the mask at {latest_output}")

    return str(latest_output) if latest_output else None, mask_vis  # Return the output image path

def select_image():
    """Opens a file dialog to select an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.dcm")])
    if file_path:
        ext = Path(file_path).suffix
        if ext not in [".dcm", ".png", ".jpg", ".jpeg"]:
            log.error(f"Unsupported file format: {ext}")
            return
        elif ext == ".dcm":
            ds = dcmread(file_path)
            image = ds.pixel_array.astype(np.float32)
            image = image / 65535
            # image = (image - image.min()) / (image.max() - image.min())
            og_img = image_utils.process_drr(image, neglog=True, clahe=True)
        else:
            og_img = np.array(Image.open(file_path))
            og_img = 255 - np.array(og_img)

        if len(og_img.shape) > 2 and og_img.shape[-1] > 3:
            og_img = og_img[:,:,:3]
        og_img, _ = pad_to_square(og_img)
        vis_img = Image.fromarray(og_img)
        vis_img = vis_img.resize((400, 400))  # Resize for display
        img_tk = ImageTk.PhotoImage(vis_img)
        image_label.config(image=img_tk)
        image_label.vis_image = img_tk  # Keep a reference
        image_label.file_path = file_path  # Store the file path
        image_label.og_image = og_img  # Store the original image
        image_label.output_dir = Path("output_results")
        image_label.multimasks = True
        image_label.points_image = []
        image_label.input_points = []
        image_label.prompts = []

def update_display_image(mask_vis):
    """Updates the displayed image in the GUI with the segmentation result."""
    img, _ = pad_to_square(mask_vis)
    img = Image.fromarray(img)
    img = img.resize((400, 400))  # Resize for display
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

def run_segmentation():
    """Runs segmentation and updates displayed image."""

    user_prompt = prompt_entry.get()
    if not user_prompt:
        result_label.config(text="Please enter a prompt.")
        return

    if hasattr(image_label, "file_path"):
        input_path = image_label.file_path
        output_dir = image_label.output_dir

        # split the prompts with commas
        user_prompt = user_prompt.split(",")
        user_prompt = [p.strip() for p in user_prompt]
        if len(user_prompt) > 1:
            image_label.multimasks = False
        multimasks = image_label.multimasks

        image_label.prompts = user_prompt

        _, mask_vis = segment(
            prompt=user_prompt,
            input_path=input_path,
            output_dir=output_dir,
            size=None,
            multimasks=multimasks
        )

        update_display_image(mask_vis)

        if len(user_prompt) > 1:
            result_label.config(text=f"Processing completed.")
        else:
            result_label.config(text=f"Processing completed. Click on the image to refine the mask.")
    else:
        result_label.config(text="Please select an image first.")

def on_image_click(event):
    """Handles user clicking on the image to select a mask."""

    prompts = image_label.prompts

    if len(prompts) > 1:
        result_label.config(text=f"Cannot select mask for multiple prompts. Please type a single prompt if you want to refine its mask.")
        return

    mask_x, mask_y = event.x, event.y
    img_h, img_w = image_label.og_image.shape[:2]
    point_x = mask_x * img_w / 400
    point_y = mask_y * img_h / 400
    points = [(point_x, point_y)]*3

    _, mask_vis = segment(
            prompt=prompts,
            points=points,
            input_path=image_label.file_path,
            output_dir=image_label.output_dir,
            size=None,
            multimasks= False
        )

    update_display_image(mask_vis)

    result_label.config(text=f"Processing completed. Click on the image to refine the mask.")


    # if len(selected_masks) > 0:
    #     # display the image with the selected mask
    #     mask_vis = image_utils.draw_masks(np.array(image_label.og_image), selected_masks, threshold=0.5, contour_thickness=3)
    #     img = Image.fromarray(mask_vis)
    #     image_tk = ImageTk.PhotoImage(img)
    #     image_label.config(image=image_tk)
    #     image_label.image = image_tk
    #     result_label.config(text="Mask selected successfully!")
    # else:
    #     result_label.config(text="No mask found at clicked location.")

# -------------------------- COMMAND-LINE ARGUMENTS --------------------------

parser = argparse.ArgumentParser(description="FluoroSAM GUI for Image Segmentation")
parser.add_argument(
    "--ckpt-path",
    type=str,
    required=True,
    help="Path to the checkpoint file.",
)
args = parser.parse_args()
ckpt_path = args.ckpt_path  # Store the checkpoint path globally

# -------------------------- CREATE GUI --------------------------

root = tk.Tk()
root.title("FluoroSAM Segmentation")
root.geometry("900x700")

# Select Image Button
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Display Image
image_label = tk.Label(root)
image_label.pack()
image_label.bind("<Button-1>", on_image_click)  # Bind click event

# Function to remove placeholder text when user clicks in the box
def on_entry_click(event):
    if prompt_entry.get() == "Enter segmentation prompt here...":
        prompt_entry.delete(0, "end")  # Clear the entry field
        prompt_entry.config(fg="white")  # Change text color to black

# Function to restore placeholder if the user leaves the box empty
def on_focus_out(event):
    if prompt_entry.get() == "":
        prompt_entry.insert(0, "Enter segmentation prompt here...")
        prompt_entry.config(fg="gray")  # Change text color to gray

# Input Prompt Entry
prompt_entry = tk.Entry(root, width=60, font=("Helvetica", 12), fg="gray")
prompt_entry.pack(pady=10)
prompt_entry.insert(0, "Enter segmentation prompt here...")

# Bind focus events
prompt_entry.bind("<FocusIn>", on_entry_click)
prompt_entry.bind("<FocusOut>", on_focus_out)

# Run Segmentation Button
process_button = tk.Button(root, text="Run Segmentation", command=run_segmentation)
process_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()

