import numpy as np
import os
import skimage as sk
import skimage.io as skio
from skimage.util import img_as_ubyte
import cv2

gaussianKernel = np.array([[1, 2, 1],[ 2, 4, 2], [1, 2, 1]])
gaussianScale = 16.0

gaussianKernelBig = np.array([[1,  4,  6,  4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1,  4,  6,  4, 1]], dtype=float) / 256.0

def l2_norm(image1, image2):
    """Compute the L2 norm between two images."""
    return -np.linalg.norm(image1 - image2)

def zncc(image1, image2, eps: float = 1e-12):
    """Compute the normalized cross-correlation between two images."""
    # Normalize each vector
    image1_flat = image1.ravel()
    image2_flat = image2.ravel()
    norm1 = (image1_flat - image1_flat.mean())
    norm2 = (image2_flat - image2_flat.mean())
    denoms = np.linalg.norm(norm1) * np.linalg.norm(norm2) + eps

    return np.dot(norm1, norm2) / denoms

def convolve(img, kernel, pad=False, mode='constant'):
    kh, kw = kernel.shape
    # do zero padding to preserve image shape
    if pad:
        pad_width = ((kh - 1) // 2, (kw - 1) // 2)
        img = np.pad(img, pad_width=(pad_width, pad_width), mode=mode)

    patches = np.lib.stride_tricks.sliding_window_view(img, (kh, kw))
    return np.einsum('ijkl,kl->ij', patches, kernel)

def gaussianFilter(img, mode='reflect', pad=False):
    return convolve(img, gaussianKernelBig, pad=pad, mode=mode)

def overlap_views(img1, img2, dy, dx):
    """Return two cropped views that overlap when img2 is shifted by (dy, dx)."""
    H, W = img1.shape
    y0 = max(0,  dy); y1 = min(H, H + dy)
    x0 = max(0,  dx); x1 = min(W, W + dx)

    v1 = img1[y0:y1, x0:x1]
    v2 = img2[y0-dy:y1-dy, x0-dx:x1-dx]
    return v1, v2

def cal_metric(metric_func, img1, img2, dy, dx):
    im1, im2 = overlap_views(img1, img2, dy, dx)
    return metric_func(im1, im2)

# fast version
def image_pyramid_align_fast(window, img1, img2, levels, metric_func=l2_norm):
    """Align img2 to img1 using image pyramid approach."""
    default_window = (5,5)
    # corase level search
    if levels == 1:
        print(img1.shape)
        print(f'window {window}')
        curr_best_index_h, curr_best_index_w = align_index(window, img1, img2, metric_func)
        print(f"{curr_best_index_h} , {curr_best_index_w}")
        return curr_best_index_h, curr_best_index_w
    else:
        filter_img1 = cv2.resize(img1, dsize=None, fx=0.5, fy=0.5)
        filter_img2 = cv2.resize(img2, dsize=None, fx=0.5, fy=0.5)

        prev_best_index_h, prev_best_index_w = image_pyramid_align_fast(window, filter_img1, filter_img2, levels-1, metric_func=metric_func)
        aligned_img2 = np.roll(img2, shift=(prev_best_index_h*2, prev_best_index_w*2), axis=(0, 1))

        curr_best_index_h, curr_best_index_w = align_index(default_window, img1, aligned_img2, metric_func=metric_func)
        print(f"{prev_best_index_h*2 + curr_best_index_h} , {prev_best_index_w*2 + curr_best_index_w}")
        return prev_best_index_h*2 + curr_best_index_h, prev_best_index_w*2 + curr_best_index_w

def image_pyramid_align(window, img1, img2, levels, metric_func=l2_norm, overlap_views=True):
    """Align img2 to img1 using image pyramid approach."""
    img1 = img1[10:img1.shape[0]-10, 10:img1.shape[1]-10]
    img2 = img2[10:img2.shape[0]-10, 10:img2.shape[1]-10]
    # corase level search
    if levels == 1:
        # print(img1.shape)
        # print(f'window {window}')
        curr_best_index_h, curr_best_index_w = align_index(window, img1, img2, metric_func, overlap_views)
        # print(f"{curr_best_index_h} , {curr_best_index_w}")
        return curr_best_index_h, curr_best_index_w
    else:
        filter_img1 = gaussianFilter(img1, pad=True, mode='reflect')
        filter_img1 = filter_img1[::2, ::2] # downsample
        filter_img2 = gaussianFilter(img2, pad=True, mode='reflect')
        filter_img2 = filter_img2[::2, ::2] # downsample

        prev_best_index_h, prev_best_index_w = image_pyramid_align((window[0]*2, window[1]*2), filter_img1, filter_img2, levels-1, 
                                                                   metric_func=metric_func, overlap_views=overlap_views)
        
        aligned_img2 = np.roll(img2, shift=(prev_best_index_h*2, prev_best_index_w*2), axis=(0, 1))

        curr_best_index_h, curr_best_index_w = align_index(window, img1, aligned_img2, 
                                                           metric_func=metric_func, 
                                                           overlap_views=overlap_views)
        
        # print(f"{prev_best_index_h*2 + curr_best_index_h} , {prev_best_index_w*2 + curr_best_index_w}")
        return prev_best_index_h*2 + curr_best_index_h, prev_best_index_w*2 + curr_best_index_w
    
def align_img(img_path):
    # name of the input file
    imname = img_path

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    ag = align((30, 30), b, g, metric_func=zncc)
    ar = align((30, 30), b, r, metric_func=zncc)
    im_out = np.dstack([ar, ag, b])

    skio.imshow(im_out)
    skio.show()

    filename, ext = os.path.splitext(os.path.basename(imname))
    fname = f'./output/{filename}_align.jpg'
    skio.imsave(fname, img_as_ubyte(im_out))