import os
import torch
import torch.nn.functional as F
import numpy as np
from math import log10
from PIL import Image
from scipy import ndimage
from scipy import signal
from scipy import linalg
import torchvision.models as models
from torch.nn.functional import adaptive_avg_pool2d
from lpips import LPIPS

from models.utils import _parse_dinov2_model_name, _make_dinov2_model, pad_tensor

MAMA_CKPT = "data/ckpt/2024_05_12_10_47_09_lora_gpt_structural_simclr_side_symm_local_symm/mama_embed_pretrained_40k_steps_last_dinov2_vit_ckpt.pth"

"""
Implementation of image quality metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
"""
def psnr(img1, img2, max_val=255):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        max_val (float): Maximum value of the images (default: 1.0)
        
    Returns:
        float: PSNR value
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.float().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.float().cpu().numpy()
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Calculate mean squared error
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    return 20 * log10(max_val) - 10 * np.log10(mse).item()


def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array
    
    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where
    
    A = 1/(2*pi*sigma^2)
    
    as define by Wolfram Mathworld 
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.float().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.float().cpu().numpy()
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    
    ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
           (sigma1_sq + sigma2_sq + C2))
    ssim = ssim.mean()
    if cs_map:
        return ssim, (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    else:
        
        return ssim


class FrechetInceptionDistance(torch.nn.Module):
    """
    Implementation of Fréchet Inception Distance (FID) for evaluating the quality of generated images.
    
    FID measures the distance between the feature representations of real and generated images
    using a pretrained Inception V3 model.
    
    Lower FID values indicate higher quality and more realistic generated images.
    """
    def __init__(self, dims=2048, model_type='inception_v3', use_torch=True):
        """
        Initialize the FID calculation module.
        
        Args:
            dims (int): Dimensionality of Inception features to use (default: 2048)
            model_type (str): Model to use ('inception_v3', 'resnet50', 'mama_vit') (default: 'inception_v3')
            use_torch (bool): Whether to use PyTorch's implementation or SciPy (default: True)
        """
        super(FrechetInceptionDistance, self).__init__()
        self.dims = dims
        self.model_type = model_type
        self.use_torch = use_torch
        
        # Initialize the model
        if model_type == 'inception_v3':
            self.model = models.inception_v3(pretrained=True)
            # Remove the final classification layer
            self.model.fc = torch.nn.Identity()
            # Set to feature extraction mode
            self.model.eval()
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model.fc = torch.nn.Identity()
            self.model.eval()
        elif model_type == 'mama_vit':
            arch_name, pretrained, num_register_tokens, patch_size = _parse_dinov2_model_name('dinov2_vitb14_reg')
            self.model = _make_dinov2_model(
                arch_name=arch_name,
                patch_size=patch_size,
                pretrained=pretrained,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                grad_ckpt=False,
            )
            state_dict = torch.load(MAMA_CKPT)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _get_activations(self, images):
        """
        Get feature activations from the model.
        
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
            
        Returns:
            torch.Tensor: Feature activations (B, dims)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Preprocess images for the model
        if self.model_type == 'mama_vit' or self.model_type == 'dinov2':
            # Pad images to required size for ViT
            images = pad_tensor(images, 518, 518)
            
            with torch.no_grad():
                features = self.model(images)
                
        elif self.model_type == 'inception_v3':
            # Resize images to 299x299 for Inception
            if images.shape[2] != 299 or images.shape[3] != 299:
                images = torch.nn.functional.interpolate(images, size=(299, 299), 
                                                         mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                features = self.model(images)
                
        else:  # Other models like ResNet
            with torch.no_grad():
                features = self.model(images)
                
        # Ensure features have the right shape
        if len(features.shape) > 2:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
            features = features.view(features.size(0), -1)
            
        return features
    
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate the Fréchet distance between two multivariate Gaussians.
        
        Args:
            mu1 (torch.Tensor/ndarray): Mean of first distribution
            sigma1 (torch.Tensor/ndarray): Covariance of first distribution
            mu2 (torch.Tensor/ndarray): Mean of second distribution
            sigma2 (torch.Tensor/ndarray): Covariance of second distribution
            eps (float): Small constant for numerical stability
            
        Returns:
            float: Fréchet distance
        """
        if self.use_torch:
            # PyTorch implementation
            diff = mu1 - mu2
            
            # Product might be almost singular
            covmean, _ = self._sqrtm(sigma1.mm(sigma2), eps=eps)
            
            # Numerical error might give small imaginary component
            if torch.is_complex(covmean):
                covmean = torch.real(covmean)
                
            tr_covmean = torch.trace(covmean)
            
            return (diff.dot(diff) + torch.trace(sigma1) + 
                    torch.trace(sigma2) - 2 * tr_covmean)
        else:
            # NumPy implementation
            mu1 = mu1.cpu().numpy() if torch.is_tensor(mu1) else mu1
            mu2 = mu2.cpu().numpy() if torch.is_tensor(mu2) else mu2
            sigma1 = sigma1.cpu().numpy() if torch.is_tensor(sigma1) else sigma1
            sigma2 = sigma2.cpu().numpy() if torch.is_tensor(sigma2) else sigma2
            
            diff = mu1 - mu2
            
            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            # Numerical error might give small imaginary component
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            tr_covmean = np.trace(covmean)
            
            return (diff.dot(diff) + np.trace(sigma1) + 
                    np.trace(sigma2) - 2 * tr_covmean)
    
    def _sqrtm(self, A, eps=1e-10):
        """
        PyTorch implementation of matrix square root for symmetric matrices.
        
        Args:
            A (torch.Tensor): Input matrix
            eps (float): Small constant for numerical stability
            
        Returns:
            torch.Tensor: Matrix square root of A
        """
        with torch.no_grad():
            # Get eigenvalues and eigenvectors
            D, U = torch.linalg.eigh(A)
            
            # Filter out small eigenvalues for stability
            D = torch.max(D, torch.tensor(eps, device=D.device))
            
            # Compute the square root
            D_sqrt = torch.diag(torch.sqrt(D))
            return U @ D_sqrt @ U.t(), D
    
    def calculate_statistics(self, images):
        """
        Calculate the mean and covariance statistics for a batch of images.
        
        Args:
            images (torch.Tensor): Batch of images
            
        Returns:
            tuple: (mean, covariance) statistics
        """
        activations = self._get_activations(images)
        
        # Calculate mean and covariance
        mu = torch.mean(activations, dim=0)
        
        # Center activations
        activations_centered = activations - mu
        
        # Calculate covariance
        if self.use_torch:
            sigma = (1.0 / (activations.shape[0] - 1)) * activations_centered.t() @ activations_centered
        else:
            activations_np = activations_centered.cpu().numpy()
            sigma = np.cov(activations_np, rowvar=False)
            sigma = torch.from_numpy(sigma).to(mu.device)
            
        return mu, sigma
    
    def calculate_fid(self, real_images, generated_images):
        """
        Calculate FID score between real and generated images.
        
        Args:
            real_images (torch.Tensor): Batch of real images
            generated_images (torch.Tensor): Batch of generated images
            
        Returns:
            float: FID score
        """
        # Move the model to the same device as images
        device = real_images.device
        self.model = self.model.to(device)
        
        # Calculate statistics for real and generated images
        mu_real, sigma_real = self.calculate_statistics(real_images)
        mu_gen, sigma_gen = self.calculate_statistics(generated_images)
        
        # Calculate FID
        fid_value = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        # Return as float (not tensor)
        return fid_value.item() if torch.is_tensor(fid_value) else float(fid_value)
    
    def forward(self, real_images, generated_images):
        """
        Forward method to calculate FID score.
        
        Args:
            real_images (torch.Tensor): Batch of real images
            generated_images (torch.Tensor): Batch of generated images
            
        Returns:
            float: FID score
        """
        return self.calculate_fid(real_images, generated_images)


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss using a pretrained classification model.
    
    This loss compares features extracted from a pretrained network (e.g., VGG)
    to measure perceptual similarity between images.
    """
    def __init__(self, 
                 model_type='vgg19', 
                 layer_weights=None, 
                 use_pretrained=True, 
                 requires_grad=False):
        """
        Initialize the perceptual loss module.
        
        Args:
            model_type (str): Which model to use ('vgg16', 'vgg19') (default: 'vgg19')
            layer_weights (dict): Weights for each layer's contribution (default: None)
            use_pretrained (bool): Whether to use pretrained weights (default: True)
            requires_grad (bool): Whether to train the feature extractor (default: False)
        """
        super(PerceptualLoss, self).__init__()
        
        self.model_type = model_type
        
        import torchvision.models as models
        
        # Default layer weights if none provided
        if layer_weights is None:
            # For VGG models, using common feature layers
            if model_type.startswith('vgg'):
                self.layer_weights = {
                    '2': 0.1,   # After first conv block
                    '7': 0.1,   # After second conv block
                    '16': 0.2,  # After third conv block
                    '25': 0.4,  # After fourth conv block
                    '34': 0.2,  # After fifth conv block
                }
            elif model_type == 'mama_vit':
                self.layer_weights = {
                    'blocks.2': 0.2,
                    'blocks.5': 0.2,
                    'blocks.8': 0.3,
                    'blocks.11': 0.3,
                }
            elif model_type == 'dinov2':
                self.layer_weights = {
                    'blocks.2': 0.2,
                    'blocks.5': 0.2,
                    'blocks.8': 0.3,
                    'blocks.11': 0.3,
                }
            else:
                self.layer_weights = {
                    '0': 1.0,  # Default weight for other models
                }
        else:
            self.layer_weights = layer_weights
        
        # Load the model
        if model_type == 'vgg16':
            self.model = models.vgg16(pretrained=use_pretrained).features
        elif model_type == 'vgg19':
            self.model = models.vgg19(pretrained=use_pretrained).features
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=use_pretrained)
        elif model_type == 'mama_vit':
            arch_name, pretrained, num_register_tokens, patch_size = _parse_dinov2_model_name('dinov2_vitb14_reg')
            self.model = _make_dinov2_model(
                arch_name=arch_name,
                patch_size=patch_size,
                pretrained=pretrained,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                grad_ckpt=False,
            )
            if use_pretrained:
                state_dict = torch.load(MAMA_CKPT)
                self.model.load_state_dict(state_dict, strict=True)
        elif model_type == 'dinov2':
            arch_name, pretrained, num_register_tokens, patch_size = _parse_dinov2_model_name('dinov2_vitb14_reg')
            self.model = _make_dinov2_model(
                arch_name=arch_name,
                patch_size=patch_size,
                pretrained=pretrained,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                grad_ckpt=False,
            )
        elif model_type == 'lpips':
            self.model = LPIPS(net='vgg', version='0.1')
            return
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze the model if not training
        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.eval()
        
        # Register hooks to extract features
        self.features = {}
        self.hooks = []
        
        for name, module in self.get_layers_by_names(self.model, self.layer_weights.keys()):
            hook = self._register_hook(name)
            module.register_forward_hook(hook)
            self.hooks.append(hook)
        
        # Loss function for comparing features
        self.criterion = torch.nn.MSELoss()
        
    def get_layers_by_names(self, model, layer_names):
        """Get model layers by their names."""
        result = []
        for name, module in model.named_modules():
            if name in layer_names:
                result.append((name, module))
        return result
        
    def _register_hook(self, name):
        """Create a hook for storing features."""
        def hook(module, input, output):
            self.features[name] = output
        return hook
        
    def forward(self, x, y, verbose=False):
        """
        Calculate perceptual loss between input images.
        
        Args:
            x (torch.Tensor): Input image
            y (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        if self.model_type == 'lpips':
            # For LPIPS, use the model directly
            return self.model(x, y).mean().squeeze()
        else:
            # Clear previous features
            self.features = {}
            
            # Extract features
            if self.model_type == 'mama_vit' or self.model_type == 'dinov2':
                x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
                y = F.interpolate(y, size=(518, 518), mode='bilinear', align_corners=False)

            self.model(x)
            x_features = {name: feat.clone() for name, feat in self.features.items()}
            
            self.features = {}
            self.model(y)
            y_features = {name: feat.clone() for name, feat in self.features.items()}
            
            # Calculate weighted loss
            loss = 0.0
            for name, weight in self.layer_weights.items():
                if name in x_features and name in y_features:
                    loss += weight * self.criterion(x_features[name], y_features[name])
                    if verbose:
                        print(f"Layer {name}: Loss = {loss.item()}")
            
            return loss


if __name__ == "__main__":
    # Test module for perceptual loss and FID
    import torch
    import time
    
    def test_perceptual_loss(x, y):
        print("Testing PerceptualLoss with tensors of shape (4, 3, 512, 512)")
        
        # Initialize perceptual loss module
        try:
            print("Initializing PerceptualLoss module...")
            percep_loss = PerceptualLoss(model_type='mama_vit', use_pretrained=True)
            
            # Ensure model is in eval mode
            percep_loss.model = percep_loss.model.to('cuda')
            percep_loss.model.eval()
            
            # Measure computation time
            start_time = time.time()
            
            # Compute perceptual loss
            print("Computing perceptual loss...")
            with torch.no_grad():  # No need to track gradients for testing
                loss_value = percep_loss(x, y, verbose=True)
                
            end_time = time.time()
            
            # Print results
            print(f"Perceptual loss value: {loss_value}")
            print(f"Computation time: {end_time - start_time:.2f} seconds")
            
            # Test with identical images (should give very small loss)
            print("\nTesting with identical images...")
            with torch.no_grad():
                identical_loss = percep_loss(x, x)
            print(f"Perceptual loss for identical images: {identical_loss}")
            
            print("\nPerceptualLoss test completed successfully!")
            
        except Exception as e:
            print(f"Error during PerceptualLoss test: {str(e)}")
            raise e
    
    def test_fid():
        print("Testing FrechetInceptionDistance with tensors of shape (4, 3, 256, 256)")
        
        # Create two random tensors with the specified shape
        real_images = torch.rand(4, 3, 256, 256).to('cuda')
        gen_images = torch.rand(4, 3, 256, 256).to('cuda')
        
        # Initialize FID module
        try:
            print("Initializing FID module...")
            # Try different models
            for model_type in ['inception_v3', 'resnet50']:
                print(f"\nTesting with {model_type} model")
                fid_module = FrechetInceptionDistance(model_type=model_type).to('cuda')
                
                # Measure computation time
                start_time = time.time()
                
                # Compute FID
                print(f"Computing FID score with {model_type}...")
                fid_score = fid_module(real_images, gen_images)
                    
                end_time = time.time()
                
                # Print results
                print(f"FID score: {fid_score}")
                print(f"Computation time: {end_time - start_time:.2f} seconds")
            
                # Test with identical images (should give very small FID)
                print("\nTesting with identical images...")
                identical_fid = fid_module(real_images, real_images)
                print(f"FID for identical images: {identical_fid}")
            
            print("\nFID test completed successfully!")
            
        except Exception as e:
            print(f"Error during FID test: {str(e)}")
            raise e
    
    # Run the tests
    # Create two random tensors with the specified shape
    x = torch.rand(4, 3, 512, 512).to('cuda')
    y = torch.rand(4, 3, 512, 512).to('cuda')
    print("Random Noise Perceptual Loss")
    test_perceptual_loss(x, y)
    
    img1 = Image.open("data/EMBED_1080_ROI_JPG/images/cohort_1/10307135/1.2.845.113973.3.60.1.58049512.20181033.1/1.2.841.113686.2750828170.1540974315.4874.1539/1.2.826.0.1.3680043.8.498.76306999899831959860027639201739063998_resized.jpg").convert("RGB")
    img2 = Image.open("data/EMBED_1080_ROI_JPG/images/cohort_1/10307135/1.2.845.113973.3.60.1.58049512.20181033.1/1.2.843.113686.2750828165.1540974314.4868.1717/1.2.826.0.1.3680043.8.498.23011094687376951968766461318057431509_resized.jpg").convert("RGB")
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))
    x = torch.from_numpy(np.array(img1)).permute(2, 0, 1).unsqueeze(0).to('cuda').float() / 255.0
    y = torch.from_numpy(np.array(img2)).permute(2, 0, 1).unsqueeze(0).to('cuda').float() / 255.0
    print("Real Mammography Perceptual Loss")
    test_perceptual_loss(x, y)
    
    img3 = Image.open("/home/xypb/SRE-Conv/images/set1/input.png").convert("RGB")
    img3 = img3.resize((512, 512))
    y = torch.from_numpy(np.array(img3)).permute(2, 0, 1).unsqueeze(0).to('cuda').float() / 255.0
    print("Real Mammography vs. Pathology Perceptual Loss")
    test_perceptual_loss(x, y)
    
    y = torch.rand(1, 3, 512, 512).to('cuda')
    print("Real Mammography vs Random Noise Perceptual Loss")
    test_perceptual_loss(x, y)
    
    # test_fid()

