import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


class DifferentiableAffine(nn.Module):
    def __init__(self, angle_init=0.0, tx_init=0.0, ty_init=0.0, scale_init=1.0):
        super().__init__()
        # Initialize learnable parameters
        self.angle = nn.Parameter(torch.tensor([float(angle_init)]))
        self.tx = nn.Parameter(torch.tensor([float(tx_init)]))
        self.ty = nn.Parameter(torch.tensor([float(ty_init)]))
        self.scale = nn.Parameter(torch.tensor([float(scale_init)]))

    def get_affine_matrix(self, batch_size, device):
        # Convert degrees to radians
        theta = torch.deg2rad(self.angle)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        s = 1.0 / self.scale

        # Construct 2x3 Affine Matrix: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        row1 = torch.stack([s * cos_theta, -s * sin_theta, self.tx], dim=0).squeeze()
        row2 = torch.stack([s * sin_theta, s * cos_theta, self.ty], dim=0).squeeze()

        # Shape (B, 2, 3)
        matrix = (
            torch.stack([row1, row2], dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(device)
        )
        return matrix

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        theta_matrix = self.get_affine_matrix(B, device)

        # Pytorch grid sampling
        grid = F.affine_grid(theta_matrix, x.size(), align_corners=True)
        warped_x = F.grid_sample(
            x, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        )

        return warped_x, theta_matrix


class ProjectedEMDLoss(nn.Module):
    def __init__(self, axis=2, sigma=2.0):
        # axis=2 represents width (columns) for (B,C,H,W)
        super().__init__()
        self.axis = axis
        self.sigma = sigma

        if self.sigma > 0:
            k_size = int(2 * 4 * self.sigma + 1)
            if k_size % 2 == 0:
                k_size += 1
            self.pad_w = k_size // 2

            # Create kernel
            x = torch.arange(k_size, dtype=torch.float32) - k_size // 2
            kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
            kernel = kernel / kernel.sum()

            # Register buffer so it moves to device automatically
            # Shape (1, 1, 1, kW)
            self.register_buffer("kernel", kernel.view(1, 1, 1, -1))

    def forward(self, img1, img2):
        # Apply Gaussian smoothing on horizontal dimension
        if self.sigma > 0:
            C = img1.shape[1]
            # Expand kernel for depthwise conv: (C, 1, 1, kW)
            k = self.kernel.repeat(C, 1, 1, 1)

            # Apply convolution
            img1 = F.conv2d(img1, k.to(img1.device), padding=(0, self.pad_w), groups=C)
            img2 = F.conv2d(img2, k.to(img2.device), padding=(0, self.pad_w), groups=C)

        # plt.figure()
        # plt.imshow(img1[0,0].detach().cpu().numpy(), cmap='gray')
        # plt.title('Smoothed Image 1')
        # plt.show()
        # plt.figure()
        # plt.imshow(img2[0,0].detach().cpu().numpy(), cmap='gray')
        # plt.title('Smoothed Image 2')
        # plt.show()

        # Project 2D -> 1D
        proj1 = torch.sum(img1, dim=3 if self.axis == 2 else 2).squeeze(1)
        proj2 = torch.sum(img2, dim=3 if self.axis == 2 else 2).squeeze(1)

        # Normalize to probability
        prob1 = proj1 / (torch.sum(proj1, dim=-1, keepdim=True)[0] + 1e-8)
        prob2 = proj2 / (torch.sum(proj2, dim=-1, keepdim=True)[0] + 1e-8)

        # plt.figure()
        # plt.plot(prob1[0].detach().cpu().numpy(), label='MLO', color='tab:blue')
        # plt.fill_between(range(len(prob1[0])), prob1[0].detach().cpu().numpy(), alpha=0.3, color='tab:blue')
        # plt.plot(prob2[0].detach().cpu().numpy(), label='CC', color='tab:red')
        # plt.fill_between(range(len(prob2[0])), prob2[0].detach().cpu().numpy(), alpha=0.3, color='tab:red')
        # plt.xlim(0, max(len(prob1[0]), len(prob2[0])))
        # plt.xlabel("AP axis")
        # plt.ylabel("Pixel count")
        # plt.legend()

        # CDF
        cdf1 = torch.cumsum(prob1, dim=-1)
        cdf2 = torch.cumsum(prob2, dim=-1)

        # Wasserstein Distance = L1 distance between CDFs
        return torch.mean(torch.abs(cdf1 - cdf2))


def get_transformed_corners(affine_matrix, width, height):
    """
    Computes the 4 corners of the original image based on a reverse affine matrix.
    affine_matrix: (1, 2, 3) tensor
    """
    device = affine_matrix.device

    # Normalized coordinates of corners [-1, 1]
    corners_norm = torch.tensor(
        [
            [-1.0, -1.0, 1.0],  # Top-Left
            [1.0, -1.0, 1.0],  # Top-Right
            [-1.0, 1.0, 1.0],  # Bottom-Left
            [1.0, 1.0, 1.0],  # Bottom-Right
        ],
        device=device,
    ).t()  # (3, 4)

    # Invert the affine matrix to get forward transform (Src -> Dst)
    # Pytorch affine_grid uses Dst -> Src mapping usually
    M_3x3 = torch.eye(3, device=device)
    M_3x3[:2, :] = affine_matrix.squeeze(0)
    M_forward = torch.inverse(M_3x3)

    # Transform
    transformed_norm = torch.matmul(M_forward, corners_norm)
    transformed_norm = transformed_norm[:2, :] / transformed_norm[2, :]  # Divide by Z

    # Convert normalized to pixel coords
    corners_pixel = torch.zeros((4, 2), device=device)
    corners_pixel[:, 0] = ((transformed_norm[0, :] + 1) / 2) * (width - 1)
    corners_pixel[:, 1] = ((transformed_norm[1, :] + 1) / 2) * (height - 1)

    return corners_pixel


def find_optimal_alignment(mlo_tensor, cc_tensor, epochs=50, lr=0.02):
    """
    Optimizes alignment using detached tensors,
    then applies result to original tensors.
    """

    # 1. Detach inputs for the optimization loop
    # We do NOT want to backprop into the generator during alignment search
    mlo_opt = mlo_tensor.detach()
    cc_opt = cc_tensor.detach()

    # Initialize alignment module
    # Can init translation based on center of mass difference if desired
    aligner = DifferentiableAffine().to(mlo_tensor.device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=lr)
    criterion = ProjectedEMDLoss(axis=2)  # optimize horizontal alignment

    # 2. Optimization Loop
    for _ in range(epochs):
        optimizer.zero_grad()
        warped, _ = aligner(mlo_opt)
        loss = criterion(warped, cc_opt)
        loss.backward()
        optimizer.step()

    # 3. Apply to ORIGINAL tensor
    # Now we use the found parameters on the attached computational graph.
    # We fix the aligner parameters so they are constants in the main graph.
    aligner.eval()
    for param in aligner.parameters():
        param.requires_grad = False

    final_image, affine_matrix = aligner(mlo_tensor)

    # Calculate corners for reference
    _, _, H, W = mlo_tensor.shape
    corners = get_transformed_corners(affine_matrix, W, H)

    return (final_image,)


def rotate_tensor(input_tensor, angle_deg, center=(0.0, 0.0)):
    """
    Rotates a tensor image around a specific center.

    Args:
        input_tensor: Shape (B, C, H, W)
        angle_deg: Rotation angle in degrees (Counter-Clockwise)
        center: Tuple (x, y) coordinates of the rotation center.
                Values should be in normalized range [-1, 1].
                (0, 0) is the center of the image.
                (-1, -1) is top-left, (1, 1) is bottom-right.
    """
    B, C, H, W = input_tensor.shape
    device = input_tensor.device

    # 1. Convert angle to radians
    # Note: Standard rotation matrix corresponds to CCW rotation of the image content
    theta = math.radians(angle_deg)

    c = math.cos(theta)
    s = math.sin(theta)

    # 2. Define Rotation Matrix R (2x2)
    # This matrix maps Target coordinates -> Source coordinates
    rot_mat = torch.tensor([[c, s], [-s, c]], device=device)

    # 3. Handle Center of Rotation
    # Translation vector T = Center - R * Center
    cx, cy = center
    center_vec = torch.tensor([cx, cy], device=device).unsqueeze(1)  # (2, 1)

    # Calculate translation needed to keep 'center' fixed
    t = center_vec - rot_mat @ center_vec  # (2, 1)

    # 4. Construct Full Affine Matrix (2x3) -> [ R | T ]
    affine_matrix = torch.cat([rot_mat, t], dim=1)  # (2, 3)

    # Expand to batch size: (B, 2, 3)
    affine_matrix = affine_matrix.unsqueeze(0).repeat(B, 1, 1)

    # 5. Apply Transformation
    # grid maps (x_target, y_target) -> (x_source, y_source)
    grid = F.affine_grid(affine_matrix, input_tensor.size(), align_corners=True)
    rotated_output = F.grid_sample(input_tensor, grid, align_corners=True)

    return rotated_output
