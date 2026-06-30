import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from matplotlib import pylab as pylab
from models.preprocess_torch import (
    normalize_tensor_01,
    _to_numpy_img,
    threshold_background_tn,
    right_orient_mammogram_tn,
    remove_text_label_tn,
    otsu_cut_tn,
    adaptive_mask_bottom_fn_tn,
    adaptive_cut_right_fn_tn,
    adaptive_cut_top_fn_tn,
    adaptive_cut_bottom_fn_tn,
)
from models.preprocess import enhance_contrast, gaussian_blur
from models.align_pectoral_line import (
    adaptive_cut_right_fn,
    adaptive_cut_bottom_fn,
    adaptive_cut_top_fn,
    apply_canny,
    get_hough_lines,
    shortlist_lines,
    pick_line_conf,
)
from models.differentiable_affine import ProjectedEMDLoss, rotate_tensor
from models.align_utils import display_distribution_res


def final_align(
    cc_tensor, mlo_tensor, best_cc_crop=None, best_mlo_rot=None, smooth_sigma=5
):
    # Determine best crop and rotation if not provided
    if best_cc_crop is None and best_mlo_rot is None:
        cc_image = Image.fromarray(
            cc_tensor.squeeze().detach().cpu().numpy() * 255
        ).convert("L")
        mlo_image = Image.fromarray(
            mlo_tensor.squeeze().detach().cpu().numpy() * 255
        ).convert("L")
        min_emd = np.inf
        min_jsd = np.inf
        best_crop = 0
        best_rot = 0
        emd_list = []
        jsd_list = []
        crop_list = list(range(0, 128, 8))
        rot_list = list(range(-6, 7, 2))
        best_cc_image = None
        best_mlo_image = None
        for crop in crop_list:
            for rot in rot_list:
                if rot > 0:
                    # rotate anticlockwise, choose left bottom corner as anchor
                    # use cv2 to rotate around anchor
                    M = cv2.getRotationMatrix2D((0, mlo_image.size[1]), rot, 1.0)
                    expanded_image = np.zeros(
                        (mlo_image.size[1] * 2, mlo_image.size[0] * 2)
                    )
                    # place the original image at the bottom left corner
                    expanded_image[mlo_image.size[1] :, : mlo_image.size[0]] = np.array(
                        mlo_image
                    )
                    mlo_image_rotated = Image.fromarray(
                        cv2.warpAffine(
                            np.array(mlo_image),
                            M,
                            (expanded_image.shape[1], expanded_image.shape[0]),
                        )
                    )
                else:
                    # rotate anticlockwise, choose top left corner as anchor
                    M = cv2.getRotationMatrix2D((0, 0), rot, 1.0)
                    expanded_image = np.zeros(
                        (mlo_image.size[1] * 2, mlo_image.size[0] * 2)
                    )
                    # place the original image at the top left corner
                    expanded_image[: mlo_image.size[1], : mlo_image.size[0]] = np.array(
                        mlo_image
                    )
                    mlo_image_rotated = Image.fromarray(
                        cv2.warpAffine(
                            np.array(mlo_image),
                            M,
                            (expanded_image.shape[1], expanded_image.shape[0]),
                        )
                    )
                # crop cc image by crop pixels from left
                cc_image_cropped = Image.fromarray(np.array(cc_image)[:, crop:]).resize(
                    (512, 512)
                )
                # post process both images with adaptive cut top and right black areas

                mlo_image_rotated = adaptive_cut_right_fn(
                    adaptive_cut_bottom_fn(
                        adaptive_cut_top_fn(np.array(mlo_image_rotated))[0]
                    )[0]
                )[0]
                mlo_image_rotated = Image.fromarray(mlo_image_rotated).resize(
                    (512, 512)
                )
                cc_image_cropped = adaptive_cut_right_fn(
                    adaptive_cut_top_fn(np.array(cc_image_cropped))[0]
                )[0]
                cc_image_cropped = Image.fromarray(cc_image_cropped).resize((512, 512))

                res = display_distribution_res(
                    mlo_image_rotated,
                    cc_image_cropped,
                    smooth_sigma=smooth_sigma,
                    show_img=False,
                )
                emd_list.append(res["emd"])
                jsd_list.append(res["jsd"])
                if res["emd"] < min_emd:
                    min_emd = res["emd"]
                    best_crop = crop
                    best_rot = rot
                    best_cc_image = cc_image_cropped
                    best_mlo_image = mlo_image_rotated
                # if res['jsd'] < min_jsd:
                #     min_jsd = res['jsd']
                #     best_crop = crop
                #     best_rot = rot
        # print(f"Best crop: {best_crop}, Best rotation: {best_rot}. Min EMD: {min_emd}, Min JSD: {min_jsd}")
        # apply crop and rotation to mlo_tensor
        # display_distribution_res(best_mlo_image, best_cc_image, smooth_sigma=smooth_sigma, show_img=True)
        best_cc_crop = best_crop
        best_mlo_rot = best_rot

    if best_mlo_rot > 0:
        wider_tensor = torch.zeros(
            (1, 1, mlo_tensor.shape[2] * 2, mlo_tensor.shape[3] * 2),
            device=mlo_tensor.device,
        )
        wider_tensor[:, :, mlo_tensor.shape[2] :, : mlo_tensor.shape[3]] = mlo_tensor
        rotated_tensor = rotate_tensor(wider_tensor, -best_mlo_rot, (-1.0, 1.0))
    else:
        wider_tensor = torch.zeros(
            (1, 1, mlo_tensor.shape[2] * 2, mlo_tensor.shape[3] * 2),
            device=mlo_tensor.device,
        )
        wider_tensor[:, :, : mlo_tensor.shape[2], : mlo_tensor.shape[3]] = mlo_tensor
        rotated_tensor = rotate_tensor(wider_tensor, -best_mlo_rot, (-1.0, -1.0))
    cropped_cc_tensor = cc_tensor[:, :, :, best_cc_crop:]
    # Post processing
    # First convert to numpy array for computation
    cc_arr = _to_numpy_img(cropped_cc_tensor, force_uint8_calc=True)[0]
    mlo_arr = _to_numpy_img(rotated_tensor, force_uint8_calc=True)[0]
    # Then apply otsu cut and adaptive cuts
    rotated_mlo_tensor, mlo_arr = otsu_cut_tn(rotated_tensor, arr=mlo_arr)
    cropped_cc_tensor, cc_arr = otsu_cut_tn(cropped_cc_tensor, arr=cc_arr)
    rotated_mlo_tensor, _, mlo_arr = adaptive_cut_top_fn_tn(
        rotated_mlo_tensor, arr=mlo_arr
    )
    rotated_mlo_tensor, _, mlo_arr = adaptive_cut_bottom_fn_tn(
        rotated_mlo_tensor, arr=mlo_arr
    )
    rotated_mlo_tensor, _, mlo_arr = adaptive_cut_right_fn_tn(
        rotated_mlo_tensor, arr=mlo_arr
    )
    cropped_cc_tensor, _, cc_arr = adaptive_cut_top_fn_tn(cropped_cc_tensor, arr=cc_arr)
    cropped_cc_tensor, _, cc_arr = adaptive_cut_right_fn_tn(
        cropped_cc_tensor, arr=cc_arr
    )
    rotated_mlo_tensor = F.interpolate(
        rotated_mlo_tensor, size=(512, 512), mode="bilinear", align_corners=False
    )
    cropped_cc_tensor = F.interpolate(
        cropped_cc_tensor, size=(512, 512), mode="bilinear", align_corners=False
    )
    return rotated_mlo_tensor, cropped_cc_tensor, best_cc_crop, best_mlo_rot


# apply preprocessing
def mammo_ap_alignment_compute(
    mlo_tensor_orig,
    cc_tensor_orig,
    operation_dict={},
    background_threshold=0,
    criterion=None,
    smooth_sigma=5,
):
    # get each parameters from operation_dict
    pectoral_removal_degree = operation_dict.get("pectoral_removal_degree", None)
    pectoral_removal_center = operation_dict.get("pectoral_removal_center", None)
    best_cc_crop = operation_dict.get("best_cc_crop", None)
    best_mlo_rot = operation_dict.get("best_mlo_rot", None)
    cc_text_mask = operation_dict.get("cc_text_mask", None)
    mlo_text_mask = operation_dict.get("mlo_text_mask", None)
    cc_bottom_y = operation_dict.get("cc_bottom_y", None)
    mlo_bottom_y = operation_dict.get("mlo_bottom_y", None)
    cc_right_x = operation_dict.get("cc_right_x", None)
    mlo_right_x = operation_dict.get("mlo_right_x", None)
    cc_top_y = operation_dict.get("cc_top_y", None)
    mlo_top_y = operation_dict.get("mlo_top_y", None)

    # convert to grayscale tensor
    cc_tensor_orig = cc_tensor_orig.mean(dim=1, keepdim=True)
    mlo_tensor_orig = mlo_tensor_orig.mean(dim=1, keepdim=True)

    if background_threshold > 0:
        cc_tensor_orig = threshold_background_tn(
            cc_tensor_orig, threshold=background_threshold
        )
        mlo_tensor_orig = threshold_background_tn(
            mlo_tensor_orig, threshold=background_threshold
        )

    # convert to arr then conduct the operations
    # reduce the operation cnt to copy tensor from GPU to CPU
    cc_arr, _ = _to_numpy_img(cc_tensor_orig, force_uint8_calc=True)
    mlo_arr, _ = _to_numpy_img(mlo_tensor_orig, force_uint8_calc=True)

    # --- Preprocessing Steps ---
    cc_tensor, _, cc_arr = right_orient_mammogram_tn(cc_tensor_orig, arr=cc_arr)
    mlo_tensor, _, mlo_arr = right_orient_mammogram_tn(mlo_tensor_orig, arr=mlo_arr)
    cc_tensor, cc_text_mask, cc_arr = remove_text_label_tn(
        cc_tensor, return_mask=True, arr=cc_arr, mask_t=cc_text_mask
    )
    mlo_tensor, mlo_text_mask, mlo_arr = remove_text_label_tn(
        mlo_tensor, return_mask=True, arr=mlo_arr, mask_t=mlo_text_mask
    )
    cc_tensor, cc_arr = otsu_cut_tn(cc_tensor, arr=cc_arr)
    mlo_tensor, mlo_arr = otsu_cut_tn(mlo_tensor, arr=mlo_arr)
    cc_tensor, cc_bottom_y, cc_arr = adaptive_mask_bottom_fn_tn(
        cc_tensor, arr=cc_arr, breast_bottom_y=cc_bottom_y
    )
    mlo_tensor, mlo_bottom_y, mlo_arr = adaptive_mask_bottom_fn_tn(
        mlo_tensor, arr=mlo_arr, breast_bottom_y=mlo_bottom_y
    )
    cc_tensor, cc_right_x, cc_arr = adaptive_cut_right_fn_tn(
        cc_tensor, arr=cc_arr, breast_right_x=cc_right_x
    )
    mlo_tensor, mlo_right_x, mlo_arr = adaptive_cut_right_fn_tn(
        mlo_tensor, arr=mlo_arr, breast_right_x=mlo_right_x
    )
    cc_tensor, cc_top_y, cc_arr = adaptive_cut_top_fn_tn(
        cc_tensor, arr=cc_arr, breast_top_y=cc_top_y
    )
    mlo_tensor, mlo_top_y, mlo_arr = adaptive_cut_top_fn_tn(
        mlo_tensor, arr=mlo_arr, breast_top_y=mlo_top_y
    )
    cc_tensor, _ = otsu_cut_tn(cc_tensor, arr=cc_arr)
    mlo_tensor, _ = otsu_cut_tn(mlo_tensor, arr=mlo_arr)

    # Determine pectoral removal parameters if not provided
    if pectoral_removal_degree is None and pectoral_removal_center is None:
        # Use numpy for Hough line detection
        mlo_arr = mlo_tensor.squeeze().detach().cpu().numpy()
        mlo_arr = enhance_contrast(mlo_arr)
        mlo_arr = gaussian_blur(mlo_arr)
        mlo_arr = (np.array(mlo_arr) * 255).astype(np.uint8)
        mlo_canny = apply_canny(mlo_arr, mask_bottom=True, mask_right=False)
        lines = get_hough_lines(mlo_canny, verbose=False)
        W = mlo_arr.shape[1]
        shortlisted_lines = shortlist_lines(lines, image_width=W, verbose=False)
        shortlisted_lines, std = pick_line_conf(mlo_arr, shortlisted_lines)

        if shortlisted_lines:
            first_line = shortlisted_lines[0]
            x1, y1 = first_line["point1"]
            x2, y2 = first_line["point2"]
            angle = first_line["angle"]

            if x1 == 0:
                y1 = int(mlo_bottom_y)
                x1 = 0
                # go left until hit the breast boundary
                while mlo_arr[int(y2), int(x2)] == 0 and x2 > 0:
                    x2 -= 1
                # go right until hit the breast boundary
                while (
                    mlo_arr[int(y2), int(x2)] > 0
                    and x2 < np.array(mlo_arr).shape[1] - 1
                ):
                    x2 += 1
                # recompute the angle
                angle = 90 - np.degrees(np.arctan2(y1 - y2, x2 - x1))
            else:
                y2 = int(mlo_bottom_y)
                x2 = 0
                # go left until hit the breast boundary
                while mlo_arr[int(y1), int(x1)] == 0 and x1 > 0:
                    x1 -= 1
                # go right until hit the breast boundary
                while (
                    mlo_arr[int(y1), int(x1)] > 0
                    and x1 < np.array(mlo_arr).shape[1] - 1
                ):
                    x1 += 1
                angle = 90 - np.degrees(np.arctan2(y2 - y1, x1 - x2))
            if x1 == 0:
                center = (x1, y1)
            elif x2 == 0:
                center = (x2, y2)
            elif y1 == 0:
                center = (x1, y1)
            elif y2 == 0:
                center = (x2, y2)
            else:
                center = (mlo_arr.shape[1] // 2, mlo_arr.shape[0] // 2)

            # print(angle, center, mlo_arr.shape)
            pectoral_removal_degree = angle
            pectoral_removal_center = center
        else:
            pectoral_removal_degree = 0
            pectoral_removal_center = (0, 0)

    if pectoral_removal_degree != 0:
        relative_center = (
            2 * (pectoral_removal_center[0] / mlo_tensor.shape[3]) - 1,
            2 * (pectoral_removal_center[1] / mlo_tensor.shape[2]) - 1,
        )

        wider_tensor = torch.zeros(
            (1, 1, mlo_tensor.shape[2], mlo_tensor.shape[3] * 2),
            device=mlo_tensor.device,
        )
        wider_tensor[:, :, :, : mlo_tensor.shape[3]] = mlo_tensor

        rotated_image = rotate_tensor(
            wider_tensor, -pectoral_removal_degree, relative_center
        )

        rotated_image, _ = otsu_cut_tn(rotated_image)
    else:
        rotated_image, _ = otsu_cut_tn(mlo_tensor)

    cc_tensor = torch.nn.functional.interpolate(
        cc_tensor, size=(512, 512), mode="bilinear", align_corners=False
    )
    # mlo_tensor = torch.nn.functional.interpolate(mlo_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    rotated_tensor = torch.nn.functional.interpolate(
        rotated_image, size=(512, 512), mode="bilinear", align_corners=False
    )

    final_mlo_tensor, final_cc_tensor, best_cc_crop, best_mlo_rot = final_align(
        cc_tensor,
        rotated_tensor,
        best_cc_crop=best_cc_crop,
        best_mlo_rot=best_mlo_rot,
        smooth_sigma=smooth_sigma,
    )

    if criterion is not None:
        emd_loss = criterion(final_mlo_tensor, final_cc_tensor)
    else:
        emd_loss = torch.tensor(0.0)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(final_cc_tensor.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.imshow(mlo_tensor.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.imshow(final_mlo_tensor.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.show()

    operation_dict = {
        "pectoral_removal_degree": pectoral_removal_degree,
        "pectoral_removal_center": pectoral_removal_center,
        "best_cc_crop": best_cc_crop,
        "best_mlo_rot": best_mlo_rot,
        "cc_text_mask": cc_text_mask,
        "mlo_text_mask": mlo_text_mask,
        "cc_bottom_y": cc_bottom_y,
        "mlo_bottom_y": mlo_bottom_y,
        "cc_right_x": cc_right_x,
        "mlo_right_x": mlo_right_x,
        "cc_top_y": cc_top_y,
        "mlo_top_y": mlo_top_y,
    }

    return final_mlo_tensor, final_cc_tensor, emd_loss, operation_dict
