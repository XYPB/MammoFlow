import torch
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt


def _to_numpy_img(tensor, force_uint8_calc=True):
    """
    Helper: Converts tensor to (H,W) numpy array for CV2 calculation.
    """
    t = tensor.detach().cpu()
    # Squeeze batch/channel dims to get 2D image
    while t.ndim > 2:
        t = t.squeeze(0)
    arr = t.numpy()

    # Normalize/Scale logic matches original preprocess.py
    is_float_01 = False
    if arr.dtype.kind == "f" and arr.max() <= 1.5:
        is_float_01 = True

    if force_uint8_calc:
        if is_float_01:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    return arr, is_float_01


def normalize_tensor_01(tensor_img):
    """
    Normalizes a tensor image to 0-1 range.
    """
    x = tensor_img.float()
    min_val = x.min()
    max_val = x.max()
    val_range = max_val - min_val

    if val_range < 1e-6:
        return tensor_img

    x_norm = (x - min_val) / val_range
    return x_norm


def threshold_background_tn(tensor_img, threshold=0):
    """
    Applies threshold background masking on a tensor.
    If input is float 0-1, threshold is assumed to be 0-255 scale (per original code behavior).
    """
    is_float = tensor_img.max() <= 1.05

    # Original logic: Image > threshold (where threshold is usually ~10-15 in 0-255 range)
    if is_float:
        # Scale threshold to 0-1
        mask = tensor_img > (threshold / 255.0)
    else:
        mask = tensor_img > threshold

    # Apply mask
    return torch.where(mask, tensor_img, torch.zeros_like(tensor_img))


def right_orient_mammogram_tn(tensor_img, arr=None):
    """
    Checks orientation and flips tensor if necessary.
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img)
    H, W = arr.shape

    # Count non-zero pixels in left vs right half
    left_nonzero = np.count_nonzero(arr[:, 0 : W // 2])
    right_nonzero = np.count_nonzero(arr[:, W // 2 :])

    is_flipped = left_nonzero < right_nonzero

    out_tensor = tensor_img
    if is_flipped:
        # Flip along width dimension (last dimension)
        out_tensor = torch.flip(tensor_img, dims=[-1])
        arr = arr[:, ::-1]

    return out_tensor, is_flipped, arr


def remove_text_label_tn(
    tensor_img, background_threshold=0, return_mask=False, arr=None, mask_t=None
):
    """
    Generates text mask via CV2 connected components -> applies to Tensor.
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=True)

    if mask_t is None:
        # --- numpy/cv2 mask generation ---
        if arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)  # Convert to 8-bit if not already
        binary_image = (arr > background_threshold).astype(np.uint8) * 255
        blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 2.0)
        binary_image = (blurred_image > background_threshold).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        mask = np.ones_like(arr, dtype=np.float32)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep large components, remove small text
            if area < 1e4 and area < np.max(stats[:, cv2.CC_STAT_AREA]):
                x, y, w, h = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT],
                )
                mask[y : y + h, x : x + w] = 0.0

        if np.sum(mask) == 0:
            mask[:] = 1.0

        # --- Apply to Tensor ---
        mask_t = torch.from_numpy(mask).to(tensor_img.device, tensor_img.dtype)

        # Handle broadcasting (e.g. if tensor is (1,1,H,W) and mask is (H,W))
        while mask_t.ndim < tensor_img.ndim:
            mask_t = mask_t.unsqueeze(0)
    else:
        mask = mask_t.squeeze().detach().cpu().numpy()

    if return_mask:
        return tensor_img * mask_t, mask_t.clone(), arr * mask
    else:
        return tensor_img * mask_t, arr * mask


def otsu_cut_tn(tensor_img, arr=None):
    """
    Calculates Otsu bounding box -> Crops Tensor.
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=True)

    # print(arr.dtype, arr.min(), arr.max())
    arr = arr.astype(np.uint8)

    # Algorithm from preprocess.py
    median = np.median(arr)
    _, thresh = cv2.threshold(arr, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = thresh
    if mask.size == 0 or not np.any(mask):
        return tensor_img, arr

    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Slicing tensor (PyTorch supports slicing just like NumPy)
    return (
        tensor_img[..., min_row : max_row + 1, min_col : max_col + 1],
        arr[min_row : max_row + 1, min_col : max_col + 1],
    )


def adaptive_mask_bottom_fn_tn(tensor_img, arr=None, breast_bottom_y=None):
    """
    Masks the bottom of the breast.
    Output: (Masked Tensor, bottom_y_coordinate)
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=False)
    # Logic works better on intensity arrays (float or int), not strictly binary mask logic

    if breast_bottom_y is None:
        y_axis_dist = np.sum(arr > 0, axis=1)
        y_axis_dist[: int(y_axis_dist.shape[0] * 0.1)] = 0
        nipple_y = np.argmax(y_axis_dist)

        y_axis_dist_slope = np.gradient(y_axis_dist)
        # plt.figure()
        # plt.plot(y_axis_dist_slope)
        # plt.title('Y-axis Slope of Edge Pixels')
        # plt.xlabel('Row Index')
        # plt.ylabel('Slope of Sum of Edge Pixels')
        # plt.show()
        y_axis_dist_slope[:nipple_y] = 0

        breast_bottom_y_candidate = np.argsort(y_axis_dist_slope)[:5]
        breast_bottom_y = np.max(breast_bottom_y_candidate)

        if y_axis_dist[breast_bottom_y] > 0.25 * arr.shape[1]:
            breast_bottom_y = arr.shape[0] - 1

    # Apply mask to tensor
    out_tensor = tensor_img.clone()
    out_tensor[..., breast_bottom_y:, :] = 0
    arr[breast_bottom_y:, :] = 0

    return out_tensor, breast_bottom_y, arr


def adaptive_cut_bottom_fn_tn(tensor_img, arr=None):
    """
    Cut the bottom of the breast.
    Output: (Masked Tensor, bottom_y_coordinate)
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=False)
    # Logic works better on intensity arrays (float or int), not strictly binary mask logic

    y_axis_dist = np.sum(arr > 0, axis=1)
    y_axis_dist[: int(y_axis_dist.shape[0] * 0.1)] = 0
    nipple_y = np.argmax(y_axis_dist)

    y_axis_dist_slope = np.gradient(y_axis_dist)
    # plt.figure()
    # plt.plot(y_axis_dist_slope)
    # plt.title('Y-axis Slope of Edge Pixels')
    # plt.xlabel('Row Index')
    # plt.ylabel('Slope of Sum of Edge Pixels')
    # plt.show()
    y_axis_dist_slope[:nipple_y] = 0

    breast_bottom_y_candidate = np.argsort(y_axis_dist_slope)[:5]
    breast_bottom_y = np.max(breast_bottom_y_candidate)

    if y_axis_dist[breast_bottom_y] > 0.25 * arr.shape[1]:
        breast_bottom_y = arr.shape[0] - 1

    # Apply cut to tensor
    out_tensor = tensor_img.clone()[..., :breast_bottom_y, :]
    arr = arr[:breast_bottom_y, :]

    return out_tensor, breast_bottom_y, arr


def adaptive_cut_right_fn_tn(tensor_img, arr=None, breast_right_x=None):
    """
    Crops right side of tensor.
    Output: (Cropped Tensor, right_x_coordinate)
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=False)

    if breast_right_x is None:
        x_axis_dist = np.sum(arr > 0, axis=0)
        if np.sum(x_axis_dist < 10) < 0.01 * x_axis_dist.shape[0]:
            return tensor_img, tensor_img.shape[-1] - 1, arr

        try:
            x_axis_dist_slope = np.gradient(x_axis_dist)
            x_axis_dist_slope[x_axis_dist > np.max(x_axis_dist) * 0.5] = 0
        except Exception as e:
            print("Error in adaptive_cut_top_fn_tn:", e)
            print("x_axis_dist:", x_axis_dist, "Image shape:", arr.shape)
            return tensor_img, tensor_img.shape[-1], arr

        breast_right_x_candidates = np.argsort(x_axis_dist_slope)[:5]
        breast_right_x = np.max(breast_right_x_candidates)

    return tensor_img[..., :breast_right_x], breast_right_x, arr[:, :breast_right_x]


def adaptive_cut_top_fn_tn(tensor_img, threshold=0.2, arr=None, breast_top_y=None):
    """
    Crops top of tensor.
    Output: (Cropped Tensor, top_y_coordinate)
    """
    if arr is None:
        arr, _ = _to_numpy_img(tensor_img, force_uint8_calc=False)

    if breast_top_y is None:
        y_axis_dist = np.sum(arr > 0, axis=1)
        y_axis_dist = gaussian_filter1d(y_axis_dist, sigma=10)
        y_axis_dist[-int(y_axis_dist.shape[0] * 0.5) :] = 0

        try:
            nipple_y = np.argmax(y_axis_dist)
            y_axis_dist_slope = np.gradient(y_axis_dist)
            y_axis_dist_slope[nipple_y:] = 0
        except Exception as e:
            print("Error in adaptive_cut_top_fn_tn:", e)
            print("y_axis_dist:", y_axis_dist, "Image shape:", arr.shape)
            return tensor_img, 0, arr

        y_axis_dist_slope[y_axis_dist > np.max(y_axis_dist) * threshold] = 0
        y_axis_dist_slope = gaussian_filter1d(y_axis_dist_slope, sigma=10)

        breast_top_y_candidates = np.argsort(y_axis_dist_slope)[-5:]
        breast_top_y = np.min(breast_top_y_candidates)

        if (
            y_axis_dist[breast_top_y] > threshold * arr.shape[1]
            or y_axis_dist_slope[breast_top_y] < 2
        ):
            breast_top_y = 0

    return tensor_img[..., breast_top_y:, :], breast_top_y, arr[breast_top_y:, :]


def enhance_contrast_tn(tensor_img, bins=256):
    """
    Differentiable global histogram equalization.
    Retains the gradient in the input.
    Operates on the entire input tensor as a single distribution.
    """
    # Clone to avoid in-place modification
    x = tensor_img.clone()

    # 0. Handle types and shapes
    is_large_scale = False
    if not x.is_floating_point():
        x = x.float()
        if x.max() > 1.05:
            is_large_scale = True
            x = x / 255.0
    elif x.max() > 1.05:
        is_large_scale = True
        x = x / 255.0

    min_val = x.min()
    max_val = x.max()
    val_range = max_val - min_val

    if val_range < 1e-6:
        return tensor_img

    # 1. Normalize to 0-1 for binning
    x_norm = (x - min_val) / val_range

    # 2. Soft Binning (Linear Interpolation)
    scaled_x = x_norm * (bins - 1)

    lower_idx = torch.floor(scaled_x).long()
    upper_idx = lower_idx + 1
    upper_idx = torch.clamp(upper_idx, max=bins - 1)

    weight_upper = scaled_x - lower_idx.float()
    weight_lower = 1.0 - weight_upper

    flattened_weights_lower = weight_lower.view(-1)
    flattened_weights_upper = weight_upper.view(-1)
    flattened_lower_idx = lower_idx.view(-1)
    flattened_upper_idx = upper_idx.view(-1)

    hist = torch.zeros(bins, device=x.device, dtype=x.dtype)
    hist.scatter_add_(0, flattened_lower_idx, flattened_weights_lower)
    hist.scatter_add_(0, flattened_upper_idx, flattened_weights_upper)

    # 3. Compute CDF
    pdf = hist / hist.sum()
    cdf = torch.cumsum(pdf, dim=0)
    cdf = cdf / cdf[-1]

    # 4. Map values
    val_lower = cdf[lower_idx]
    val_upper = cdf[upper_idx]

    y = val_lower * weight_lower + val_upper * weight_upper

    # 5. Restore scale if needed
    if is_large_scale:
        y = y * 255.0

    return y


def gaussian_blur_tn(tensor_img, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian blur to a tensor image using PyTorch convolution with Gaussian kernel.

    Args:
        tensor_img: Input tensor of shape (H, W), (C, H, W), or (B, C, H, W)
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred tensor with same shape as input
    """
    x = tensor_img.float()
    original_shape = x.shape
    original_ndim = x.ndim

    # Reshape to (B, C, H, W) for convolution
    while x.ndim < 4:
        x = x.unsqueeze(0)

    batch_size, channels, height, width = x.shape

    # Create Gaussian kernel
    kernel = _create_gaussian_kernel(kernel_size, sigma, device=x.device, dtype=x.dtype)

    # Expand kernel for all channels (depthwise convolution)
    kernel = kernel.repeat(channels, 1, 1, 1)

    # Apply convolution with padding to maintain spatial dimensions
    pad = kernel_size // 2
    blurred = torch.nn.functional.conv2d(x, kernel, padding=pad, groups=channels)

    # Restore original shape
    blurred = blurred.view(original_shape)

    return blurred


def _create_gaussian_kernel(kernel_size, sigma, device, dtype):
    """
    Creates a 2D Gaussian kernel for convolution.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        device: Device to create the kernel on
        dtype: Data type of the kernel

    Returns:
        Gaussian kernel of shape (1, 1, kernel_size, kernel_size)
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian
    x = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size // 2)
    gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()

    # Create 2D Gaussian by outer product
    gaussian_2d = gaussian_1d.unsqueeze(-1) @ gaussian_1d.unsqueeze(-1).t()
    gaussian_2d /= gaussian_2d.sum()

    # Reshape to (1, 1, kernel_size, kernel_size) for conv2d
    kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)

    return kernel
