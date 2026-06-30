import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pylab as pylab
from skimage.feature import canny
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter1d
from skimage.draw import polygon
from skimage.transform import hough_line, hough_line_peaks
from models.preprocess import (
    read_image,
    otsu_cut,
    right_orient_mammogram,
    enhance_contrast,
    gaussian_blur,
    remove_text_label,
    threshold_background,
)


def mask_bottom_fn(image):
    # mask out the bottom 10% of the image
    mask = np.ones(image.shape)
    mask[int(image.shape[0] * 0.9) :, :] = 0
    masked_image = image * mask
    return masked_image


def adaptive_mask_bottom_fn(image):
    y_axis_dist = np.sum(np.array(image) > 0, axis=1)
    y_axis_dist[: int(y_axis_dist.shape[0] * 0.1)] = (
        0  # Ignore the top 10% of the image
    )
    nipple_y = np.argmax(y_axis_dist)
    y_axis_dist_slope = np.gradient(y_axis_dist)
    # Find the breast bottom line as the row with the steepest negative slope and below the nipple
    y_axis_dist_slope[:nipple_y] = 0  # Ignore slopes above the nipple
    # pick the lowest position to avoid error
    breast_bottom_y_candidate = np.argsort(y_axis_dist_slope)[:5]
    breast_bottom_y = np.max(breast_bottom_y_candidate)
    # double check to ensure the intensity on the bottom line is low
    # print(y_axis_dist[breast_bottom_y], np.array(image).shape[1])
    if y_axis_dist[breast_bottom_y] > 0.25 * np.array(image).shape[1]:
        breast_bottom_y = image.shape[0] - 1  # If not, set to the bottom of the image
    # mask out the bottom part of the image
    mask = np.ones(image.shape)
    mask[breast_bottom_y:, :] = 0
    masked_image = image * mask
    return masked_image, breast_bottom_y


def adaptive_cut_top_fn(image, threshold=0.2):
    y_axis_dist = np.sum(np.array(image) > 0, axis=1)
    y_axis_dist = gaussian_filter1d(y_axis_dist, sigma=10)
    y_axis_dist[-int(y_axis_dist.shape[0] * 0.5) :] = (
        0  # Ignore the bottom 50% of the image
    )
    # plt.figure()
    # plt.plot(y_axis_dist)
    # plt.title('Y-axis Pixel Distribution for Top Cut')
    # plt.xlabel('Row Index')
    # plt.ylabel('Number of Non-Zero Pixels')
    # plt.show()
    nipple_y = np.argmax(y_axis_dist)
    y_axis_dist_slope = np.gradient(y_axis_dist)
    y_axis_dist_slope[nipple_y:] = 0  # Ignore slopes below the nipple
    # smooth the distribution slope
    y_axis_dist_slope[y_axis_dist > np.max(y_axis_dist) * threshold] = (
        0  # Ignore high intensity areas
    )
    y_axis_dist_slope = gaussian_filter1d(y_axis_dist_slope, sigma=10)
    # y_axis_dist_slope[:50] = 0
    # plt.figure()
    # plt.plot(y_axis_dist_slope)
    # plt.title('Y-axis Pixel Distribution Slope for Top Cut')
    # plt.xlabel('Row Index')
    # plt.ylabel('Slope')
    # plt.show()
    # Find the breast top line as the row with the steepest positive slope and above the nipple
    breast_top_y_candidates = np.argsort(y_axis_dist_slope)[-5:]
    breast_top_y = np.min(breast_top_y_candidates)
    if (
        y_axis_dist[breast_top_y] > threshold * np.array(image).shape[1]
        or y_axis_dist_slope[breast_top_y] < 2
    ):
        breast_top_y = 0  # If not, set to the top of the image
    # print("Breast top y-coordinate:", breast_top_y, y_axis_dist_slope[breast_top_y])
    return image[breast_top_y:, :], breast_top_y


def adaptive_cut_bottom_fn(image):
    y_axis_dist = np.sum(np.array(image) > 0, axis=1)
    y_axis_dist[: int(y_axis_dist.shape[0] * 0.1)] = (
        0  # Ignore the top 10% of the image
    )
    nipple_y = np.argmax(y_axis_dist)
    y_axis_dist_slope = np.gradient(y_axis_dist)
    # Find the breast bottom line as the row with the steepest negative slope and below the nipple
    y_axis_dist_slope[:nipple_y] = 0  # Ignore slopes above the nipple
    # pick the lowest position to avoid error
    breast_bottom_y_candidate = np.argsort(y_axis_dist_slope)[:5]
    breast_bottom_y = np.max(breast_bottom_y_candidate)
    # double check to ensure the intensity on the bottom line is low
    # print(y_axis_dist[breast_bottom_y], np.array(image).shape[1])
    if y_axis_dist[breast_bottom_y] > 0.25 * np.array(image).shape[1]:
        breast_bottom_y = image.shape[0] - 1  # If not, set to the bottom of the image
    return image[:breast_bottom_y, :], breast_bottom_y


def adaptive_cut_right_fn(image):
    # assume that image is processed with otsu_cut and remove_text_label
    x_axis_dist = np.sum(np.array(image) > 0, axis=0)
    if np.sum(x_axis_dist < 10) < 0.01 * x_axis_dist.shape[0]:
        return image, image.shape[1]
    x_axis_dist_slope = np.gradient(x_axis_dist)
    x_axis_dist_slope[x_axis_dist > np.max(x_axis_dist) * 0.5] = (
        0  # Ignore high intensity areas
    )
    breast_right_x_candidates = np.argsort(x_axis_dist_slope)[:5]  # Get top 5 peaks
    breast_right_x = np.max(breast_right_x_candidates)
    return image[:, :breast_right_x], breast_right_x


def mask_right_fn(image):
    # mask out the right 40% of the image
    mask = np.ones(image.shape)
    mask[:, int(image.shape[1] * 0.6) :] = 0
    masked_image = image * mask
    return masked_image


def apply_canny(image, mask_bottom=True, mask_right=False):
    if np.array(image).max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already
    if mask_bottom:
        # image = mask_bottom_fn(image)
        image, _ = adaptive_mask_bottom_fn(image)
    if mask_right:
        image = mask_right_fn(image)
    canny_img = canny(image, 20)
    return sobel(canny_img)


def line_box_intersection(x1, y1, x2, y2, width, height):
    """
    Find the intersection points of a line defined by two points with a rectangular box.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        width, height: Box dimensions (box goes from (0,0) to (width-1, height-1))

    Returns:
        List of intersection points [(x, y), ...] or empty list if no intersection
    """
    # Define box boundaries
    box_left, box_right = 0, width - 1
    box_top, box_bottom = 0, height - 1

    intersections = []

    # Handle vertical line case
    if x1 == x2:
        x = x1
        if box_left <= x <= box_right:
            # Find y-range of the line segment
            y_min, y_max = min(y1, y2), max(y1, y2)
            # Intersect with box y-range
            y_start = max(y_min, box_top)
            y_end = min(y_max, box_bottom)
            if y_start <= y_end:
                intersections.append((x, y_start))
                if y_start != y_end:
                    intersections.append((x, y_end))
        return intersections

    # Handle horizontal line case
    if y1 == y2:
        y = y1
        if box_top <= y <= box_bottom:
            # Find x-range of the line segment
            x_min, x_max = min(x1, x2), max(x1, x2)
            # Intersect with box x-range
            x_start = max(x_min, box_left)
            x_end = min(x_max, box_right)
            if x_start <= x_end:
                intersections.append((x_start, y))
                if x_start != x_end:
                    intersections.append((x_end, y))
        return intersections

    # General case: non-vertical, non-horizontal line
    # Line equation: y = mx + b
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Check intersection with each box edge
    potential_intersections = []

    # Left edge (x = box_left)
    y_at_left = m * box_left + b
    if box_top <= y_at_left <= box_bottom:
        potential_intersections.append((box_left, y_at_left))

    # Right edge (x = box_right)
    y_at_right = m * box_right + b
    if box_top <= y_at_right <= box_bottom:
        potential_intersections.append((box_right, y_at_right))

    # Top edge (y = box_top)
    x_at_top = (box_top - b) / m
    if box_left <= x_at_top <= box_right:
        potential_intersections.append((x_at_top, box_top))

    # Bottom edge (y = box_bottom)
    x_at_bottom = (box_bottom - b) / m
    if box_left <= x_at_bottom <= box_right:
        potential_intersections.append((x_at_bottom, box_bottom))

    # Remove duplicates (corner intersections might be counted twice)
    unique_intersections = []
    tolerance = 1e-10

    for point in potential_intersections:
        is_duplicate = False
        for existing in unique_intersections:
            if (
                abs(point[0] - existing[0]) < tolerance
                and abs(point[1] - existing[1]) < tolerance
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_intersections.append(point)

    # Filter intersections that are actually on the line segment
    # (not just on the infinite line)
    def point_on_segment(px, py):
        # Check if point (px, py) lies on line segment from (x1, y1) to (x2, y2)
        # Using parameter t: point = (1-t)*(x1,y1) + t*(x2,y2), where 0 <= t <= 1
        if abs(x2 - x1) > tolerance:
            t = (px - x1) / (x2 - x1)
        else:
            t = (py - y1) / (y2 - y1) if abs(y2 - y1) > tolerance else 0

        return 0 <= t <= 1

    # Only keep intersections that are on the line segment
    for point in unique_intersections:
        if point_on_segment(point[0], point[1]):
            intersections.append(point)

    return intersections


def shortlist_lines(lines, image_width=None, verbose=False):
    MIN_ANGLE = 10
    MAX_ANGLE = 60
    MIN_DIST = 5
    MAX_DIST = 500
    if image_width:
        W = image_width
        MIN_DIST = max(MIN_DIST, 0.01 * W)
        MAX_DIST = min(MAX_DIST, 0.80 * W)

    shortlisted_lines = [
        x
        for x in lines
        if (x["dist"] >= MIN_DIST)
        & (x["dist"] <= MAX_DIST)
        & (x["angle"] >= MIN_ANGLE)
        & (x["angle"] <= MAX_ANGLE)
    ]
    shortlisted_lines.sort(key=lambda x: x["angle"])
    if verbose:
        print("\nShorlisted lines")
        for i in shortlisted_lines:
            print(
                "Angle: {:.2f}, Dist: {:.2f}, Conf: {:.2f}".format(
                    i["angle"], i["dist"], i["conf"]
                )
            )
    return shortlisted_lines


def get_hough_lines(canny_img, verbose=False):
    h, theta, d = hough_line(canny_img)
    lines = list()
    # plt.figure()
    # plt.imshow(np.log(1 + h), extent=[np.degrees(theta[-1]), np.degrees(theta[0]), d[-1], d[0]], cmap='gray', aspect='auto')
    # plt.title('Hough Transform')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Distance (pixels)')
    # plt.colorbar(label='Log Accumulator Value')
    # plt.show()
    if verbose:
        print("\nAll hough lines")
    for accums, angle, dist in zip(
        *hough_line_peaks(
            h, theta, d, min_distance=50, min_angle=5, threshold=0.3 * np.max(h)
        )
    ):
        if verbose:
            print(
                "Angle: {:.2f}, Dist: {:.2f}, Conf: {:<.2f}".format(
                    np.degrees(angle), dist, accums / h.max()
                )
            )
        x1 = 0
        angle = max(angle, 1e-3)
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        intersections = line_box_intersection(
            x1, y1, x2, y2, canny_img.shape[1], canny_img.shape[0]
        )
        # print(intersections)
        if len(intersections) != 2:
            continue
        x1, y1 = intersections[0]
        x2, y2 = intersections[1]
        lines.append(
            {
                "dist": dist,
                "angle": np.degrees(angle),
                "point1": [x1, y1],
                "point2": [x2, y2],
                "conf": accums / h.max(),
            }
        )

    return lines


def remove_pectoral(shortlisted_lines):
    shortlisted_lines.sort(key=lambda x: x["dist"])
    pectoral_line = shortlisted_lines[0]
    d = pectoral_line["dist"]
    theta = np.radians(pectoral_line["angle"])

    x_intercept = d / np.cos(theta)
    y_intercept = d / np.sin(theta)

    return polygon([0, 0, y_intercept], [0, x_intercept, 0])


def pick_line(image, shortlist_lines):
    if len(shortlist_lines) == 0:
        return [], []
    best_line = None
    min_std = np.inf
    for line in shortlist_lines:
        rr, cc = remove_pectoral([line])
        rr = np.clip(rr, 0, image.shape[0] - 1)
        cc = np.clip(cc, 0, image.shape[1] - 1)
        # ignore background regions
        segmented_roi = image[rr, cc]
        segmented_roi = segmented_roi.flatten()[segmented_roi > 0]
        target_std = np.std(segmented_roi)
        if target_std < min_std:
            min_std = target_std
            best_line = line
    return [best_line], min_std


# Pick the line that minimizes the remaining intensity after removing the pectoral muscle at the top
def pick_line_top_remain(image, shortlist_lines):
    if len(shortlist_lines) == 0:
        return [], []
    best_line = None
    min_intensity = np.inf
    for line in shortlist_lines:
        rr, cc = remove_pectoral([line])
        rr = np.clip(rr, 0, image.shape[0] - 1)
        cc = np.clip(cc, 0, image.shape[1] - 1)
        # remove the segmented region
        segmented_img = image.copy()
        segmented_img[rr, cc] = 0
        # calculate the mean intensity of the remaining top 5% region
        top_region_intensity = np.sum(
            segmented_img[:, : int(0.05 * image.shape[0])] > 0
        )
        if top_region_intensity < min_intensity:
            min_intensity = top_region_intensity
            best_line = line
    return [best_line], min_intensity


# Pick the line that minimizes the remaining intensity after removing the pectoral muscle at the top
def pick_line_conf(image, shortlist_lines):
    if len(shortlist_lines) == 0:
        return [], []
    best_line = None
    max_conf = -np.inf
    for line in shortlist_lines:
        confidence = line.get("conf", 0)
        if confidence < 0.1:  # ignore low confidence lines
            continue
        if confidence > max_conf:
            max_conf = confidence
            best_line = line
    return [best_line], max_conf


def display_image(
    filename_pair,
    verbose=False,
    dest=None,
    show_img=True,
    filled_align=False,
    background_threshold=0,
):
    filename = filename_pair[0]
    image = read_image(filename)
    if background_threshold > 0:
        image = threshold_background(image, threshold=background_threshold)
    cc_path = filename_pair[1]
    cc_image = read_image(cc_path)
    if background_threshold > 0:
        cc_image = threshold_background(cc_image, threshold=background_threshold)
    cc_image, _ = right_orient_mammogram(cc_image)
    cc_image = remove_text_label(cc_image)
    cc_image = otsu_cut(cc_image)
    cc_image, _ = adaptive_mask_bottom_fn(np.array(cc_image) / 255)
    cc_image = adaptive_cut_right_fn(cc_image)[0]
    cc_image = adaptive_cut_top_fn(cc_image)[0]
    cc_image = otsu_cut(cc_image)
    image, is_flipped = right_orient_mammogram(image)
    image = remove_text_label(image)
    image = otsu_cut(image)
    original_mlo_image = image.copy()
    original_mlo_image, _ = adaptive_mask_bottom_fn(np.array(original_mlo_image) / 255)
    image = enhance_contrast(image)
    image = gaussian_blur(image)
    # rescale to 0-1
    image = np.array(image) / 255
    image, breast_bottom_y = adaptive_mask_bottom_fn(image)
    # if the breast_bottom_y is zero, go up
    while image[breast_bottom_y, 0] == 0 and breast_bottom_y > 0:
        breast_bottom_y -= 1
    # plt.imshow(image, cmap='gray')
    # plt.show()
    canny_image = apply_canny(np.array(image), mask_bottom=True, mask_right=False)
    # plt.imshow(canny_image, cmap='gray')
    # plt.show()
    lines = get_hough_lines(canny_image, verbose)
    W = image.shape[1]
    shortlisted_lines = shortlist_lines(lines, image_width=W, verbose=verbose)
    # shortlisted_lines, std = pick_line(image, shortlisted_lines)
    # shortlisted_lines, std = pick_line_top_remain(image, shortlisted_lines)
    shortlisted_lines, std = pick_line_conf(image, shortlisted_lines)

    if show_img:
        fig, axes = plt.subplots(1, 6, figsize=(12, 8))
        fig.tight_layout(pad=3.0)
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0])

        axes[0].set_title("Right-oriented")
        axes[0].imshow(original_mlo_image, cmap=pylab.cm.gray)
        axes[0].axis("on")

        axes[1].set_title("Hough Lines on Canny Edge")
        axes[1].imshow(canny_image, cmap=pylab.cm.gray)
        axes[1].axis("on")
        axes[1].set_xlim(0, image.shape[1])
        axes[1].set_ylim(image.shape[0])
        for line in lines:
            axes[1].plot(
                (line["point1"][0], line["point2"][0]),
                (line["point1"][1], line["point2"][1]),
                "-r",
            )

        axes[2].set_title("Shortlisted Lines")
        axes[2].imshow(canny_image, cmap=pylab.cm.gray)
        axes[2].axis("on")
        axes[2].set_xlim(0, image.shape[1])
        axes[2].set_ylim(image.shape[0])
        for line in shortlisted_lines:
            # print(line.get('conf', 0))
            # print(line)
            axes[2].plot(
                (line["point1"][0], line["point2"][0]),
                (line["point1"][1], line["point2"][1]),
                "-r",
            )

    aligned_image_pair = (None, None)
    if shortlisted_lines:
        first_line = shortlisted_lines[0]
        x1, y1 = first_line["point1"]
        x2, y2 = first_line["point2"]
        angle = first_line["angle"]
        if filled_align:
            if x1 == 0:
                y1 = int(breast_bottom_y)
                x1 = 0
                # go left until hit the breast boundary
                while image[int(y2), int(x2)] == 0 and x2 > 0:
                    x2 -= 1
                # go right until hit the breast boundary
                while image[int(y2), int(x2)] > 0 and x2 < np.array(image).shape[1] - 1:
                    x2 += 1
                # recompute the angle
                angle = 90 - np.degrees(np.arctan2(y1 - y2, x2 - x1))
                # print(x1, y1, x2, y2)
                # print(angle)
            else:
                y2 = int(breast_bottom_y)
                x2 = 0
                # go left until hit the breast boundary
                while image[int(y1), int(x1)] == 0 and x1 > 0:
                    x1 -= 1
                # go right until hit the breast boundary
                while image[int(y1), int(x1)] > 0 and x1 < np.array(image).shape[1] - 1:
                    x1 += 1
                angle = 90 - np.degrees(np.arctan2(y2 - y1, x1 - x2))
                # print(angle)
        if x1 == 0:
            center = (x1, y1)
        elif x2 == 0:
            center = (x2, y2)
        elif y1 == 0:
            center = (x1, y1)
        elif y2 == 0:
            center = (x2, y2)
        else:
            center = (image.shape[1] // 2, image.shape[0] // 2)
        # double the image width to prevent cropping during rotation
        new_width = 2 * image.shape[1]
        # Expand the image with 0s according to the new width
        expanded_image = np.zeros((image.shape[0], new_width))
        # not to use the enhanced image to preserve the original intensity
        expanded_image[:, : image.shape[1]] = original_mlo_image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotate_image = cv2.warpAffine(
            expanded_image, M, (expanded_image.shape[1], expanded_image.shape[0])
        )
        # crop the image after rotation as additional tissue may come out after rotation
        rotate_image, _ = adaptive_mask_bottom_fn(rotate_image)
        # may harm regular cases
        rotate_image, _ = adaptive_cut_top_fn(rotate_image, threshold=0.5)
        rotate_image = otsu_cut(rotate_image)
        if show_img:
            axes[3].set_title("Rotated Image")
            axes[3].imshow(rotate_image, cmap=pylab.cm.gray)
            axes[3].axis("on")
            # display the cc image
            # resize cc image to the same width as the mlo image
            cc_image = cv2.resize(
                np.array(cc_image),
                (np.array(rotate_image).shape[1], np.array(rotate_image).shape[0]),
            )
            axes[4].set_title("CC Image")
            axes[4].imshow(cc_image, cmap=pylab.cm.gray)
            axes[4].axis("on")
        aligned_image_pair = (rotate_image, cc_image)

        rr, cc = remove_pectoral(shortlisted_lines)
        rr = np.clip(rr, 0, original_mlo_image.shape[0] - 1)
        cc = np.clip(cc, 0, original_mlo_image.shape[1] - 1)
        original_mlo_image[rr, cc] = 0
        if show_img:
            original_mlo_image = otsu_cut(original_mlo_image)
            axes[5].set_title("Pectoral muscle removed")
            axes[5].imshow(original_mlo_image, cmap=pylab.cm.gray)
            axes[5].axis("on")
            if dest:
                filename = filename.split("/")[-1]
                fig_dest = (
                    os.path.join(dest, filename)
                    .replace(".png", "_vis.png")
                    .replace(".jpg", "_vis.jpg")
                )
                plt.savefig(fig_dest, bbox_inches="tight")
                filename = filename.split("/")[-1]
                pectoral_removed_dest = (
                    os.path.join(dest, filename)
                    .replace(".png", "_pectoral_removed.png")
                    .replace(".jpg", "_pectoral_removed.jpg")
                )
                if isinstance(original_mlo_image, np.ndarray):
                    if original_mlo_image.max() <= 1.0:
                        original_mlo_image = original_mlo_image * 255
                    original_mlo_image = Image.fromarray(
                        original_mlo_image.astype(np.uint8))
                original_mlo_image.save(pectoral_removed_dest)
        if dest:
            filename = filename.split("/")[-1]
            mlo_dest = (
                os.path.join(dest, filename)
                .replace(".png", "_aligned.png")
                .replace(".jpg", "_aligned.jpg")
            )
            if isinstance(rotate_image, np.ndarray):
                rotate_image = Image.fromarray((rotate_image * 255).astype(np.uint8))
            rotate_image.save(mlo_dest)
            cc_filename = cc_path.split("/")[-1]
            cc_dest = (
                os.path.join(dest, cc_filename)
                .replace(".png", "_aligned.png")
                .replace(".jpg", "_aligned.jpg")
            )
            if isinstance(cc_image, np.ndarray):
                if cc_image.max() <= 1.0:
                    cc_image = cc_image * 255
                cc_image = Image.fromarray(cc_image.astype(np.uint8))
            cc_image.save(cc_dest)
    else:
        original_mlo_image, _ = adaptive_mask_bottom_fn(original_mlo_image)
        original_mlo_image, _ = adaptive_cut_top_fn(original_mlo_image)
        original_mlo_image = otsu_cut(original_mlo_image)
        if show_img:
            axes[3].set_title("Bottom trimmed Image")
            axes[3].imshow(original_mlo_image, cmap=pylab.cm.gray)
            axes[3].axis("on")
            # display the cc image
            # resize cc image to the same width as the mlo image
            cc_image = cv2.resize(
                np.array(cc_image),
                (
                    np.array(original_mlo_image).shape[1],
                    np.array(original_mlo_image).shape[0],
                ),
            )
            axes[4].set_title("CC Image")
            axes[4].imshow(cc_image, cmap=pylab.cm.gray)
            axes[4].axis("on")
            axes[5].set_title("Pectoral muscle removal skipped")
            axes[5].imshow(np.zeros_like(original_mlo_image), cmap=pylab.cm.gray)
            axes[5].axis("on")
            if dest:
                filename = filename.split("/")[-1]
                fig_dest = (
                    os.path.join(dest, filename)
                    .replace(".png", "_vis.png")
                    .replace(".jpg", "_vis.jpg")
                )
                plt.savefig(fig_dest, bbox_inches="tight")
        aligned_image_pair = (original_mlo_image, cc_image)
        if dest:
            filename = filename.split("/")[-1]
            mlo_dest = os.path.join(dest, filename)
            # double otsu cut to remove the bottom region
            original_mlo_image = otsu_cut(original_mlo_image)
            if isinstance(original_mlo_image, np.ndarray):
                original_mlo_image = Image.fromarray(
                    (original_mlo_image * 255).astype(np.uint8)
                )
            original_mlo_image.save(mlo_dest)
            cc_filename = cc_path.split("/")[-1]
            cc_dest = (
                os.path.join(dest, cc_filename)
                .replace(".png", "_aligned.png")
                .replace(".jpg", "_aligned.jpg")
            )
            if isinstance(cc_image, np.ndarray):
                cc_image = Image.fromarray((cc_image * 255).astype(np.uint8))
            cc_image.save(cc_dest)
    if show_img:
        plt.show()
        plt.cla()
        plt.close()
    return aligned_image_pair
