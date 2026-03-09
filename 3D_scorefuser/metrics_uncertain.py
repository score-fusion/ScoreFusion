import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim_sk
import os
import argparse
import glob
from tqdm import tqdm
from sklearn import metrics
# import matplotlib.pyplot as plt
import cv2


# def make_plot(data, path, n_row=4):
#     # expected shape of data: 152 192 192
#     d,w,h = data.shape
#     print("saving plot in: ",path)
#     data = scale_data(data)
#     data[0][0][0] = 0
#     data[-1][-1][-1] = 255
    
#     canvas = np.zeros((w*n_row, h*n_row))
#     for i in range(n_row):
#         for j in range(n_row):
#             current_plot = n_row*i+j+1
#             canvas[i*w:(i+1)*w,j*h:(j+1)*h] = data[d//(n_row**2)*current_plot, :, :]
    
#     # use a larger plot window
#     plt.figure(figsize=(20, 20))
#     plt.imshow(canvas, cmap="gray")
#     plt.savefig(path + ".png")
#     plt.close()

#     max_size = np.max(data.shape)

#     canvas_3_view = np.zeros((max_size,max_size*3))
#     canvas_3_view[:w,:h] = data[d//2,:,:]
#     canvas_3_view[:d,max_size:max_size+h] = data[::-1,w//2,::-1]
#     canvas_3_view[:d,max_size*2:max_size*2+w] = data[::-1,::-1,h//2]

#     plt.figure(figsize=(20, 20))
#     plt.imshow(canvas_3_view, cmap="gray")
#     plt.savefig(path + "_3view.png")
#     plt.close()

# def viz(self, cur_images, path, video = True, plot = True, file = True):
#         # cur_images of shape 1 d w h or just d w h
#         cur_images = torch.squeeze(cur_images)
#         cur_images = cur_images.transpose(2, 0)
#         sampleImage = cur_images.cpu().numpy()
#         nifti_img = nib.Nifti1Image(sampleImage, affine=self.ref.affine)
#         if file:
#             nib.save(nifti_img, path + '.nii.gz')
#         sampleImage = np.transpose(sampleImage, (2, 1, 0))
#         if plot:
#             make_plot(sampleImage, path)
#         if video:
#             make_video(sampleImage, path)

#         return 1
    
def viz_plot(img, path = "debug"):
    # plot the 3D image in 3 planes using matplotlib
    
    path += ".png"
    print("saving plot in: ",path)
    
    img = np.array(img)
    img = np.squeeze(img)
    # print("in sale data when ploting: data min:", np.min(img), "data max:", np.max(img), "data mean:", np.mean(img))
    # data = (data - np.min(data)) / (np.abs(np.max(data) - np.min(data)) +0.001) * 255 # .0001 supports for all zero input
    data_max, data_min = 1, 0
    img = (img - data_min) / np.abs(data_max - data_min) * 255 # .0001 supports for all zero input
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # print(img.shape)
    d,w,h = img.shape
    max_size = np.max(img.shape)

    canvas_3_view = np.zeros((max_size,max_size*3))
    canvas_3_view[:w,:h] = img[d//2,:,:]
    # import pdb; pdb.set_trace()
    canvas_3_view[:d,max_size:max_size+h] = img[:,w//2,:] #img[::-1,w//2,::-1]
    canvas_3_view[:d,max_size*2:max_size*2+w] = img[::-1,::-1,h//2]
    
    # Save the image using OpenCV
    # image_bgr = cv2.cvtColor(canvas_3_view, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, canvas_3_view)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(canvas_3_view, cmap="gray")
    # plt.savefig(path + "_3view.png")
    # plt.close()
    return
    


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()



def ssim(gt, res):
    # if mask is None:
    #     return ssim_sk(gt,res,data_range = 1)
    # else:
    #     #TODO: add support for masked ssim here
    #     return ssim_sk(gt,res,data_range = 1)
    return ssim_sk(gt,res,data_range = 1)

class ResDataset(Dataset):
    def __init__(self, args):
        self.GT_paths = []
        self.inferece_paths = []
        for path_sufix in ["", "1" , "2", "3", "4" ]:
            res = args.res
            res = res.replace('*', path_sufix)
            gt_files = sorted(glob.glob(res + '/*GT*.npy'))
            inference_files = sorted(glob.glob(res + '/*inferece*.npy'))
            if len(gt_files) == 0 or len(inference_files) == 0 or len(gt_files) != len(inference_files):
                print(f"problematic files found in {res}")
                # print(gt_files),
                # print(inference_files)
                # import pdb; pdb.set_trace()
                continue
            self.GT_paths.append(gt_files)
            self.inferece_paths.append(inference_files)
        
        # print(self.GT_paths)
        self.gt_files = self.GT_paths[0]

    def __len__(self):
        return 40 #len(self.gt_files)

    def __getitem__(self, idx):
        gt_file = self.gt_files[idx]
        # try:
        res_files = [self.inferece_paths[i][idx] for i in range(len(self.inferece_paths))]

        gt_img = np.load(gt_file)
        result_imgs = [np.load(res_files[i]) for i in range((len(res_files)))] # np.load(res_file)
        
        # normalize
        max_v, min_v = np.max(gt_img), np.min(gt_img)
        # if min_v != 0 or max_v != 1:
        #     print(f"min value is not 0 or max value is not 1, {min_v}, {max_v}")
        gt_img = (gt_img - min_v)/(max_v - min_v)
        for i in range(len(result_imgs)):
            result_imgs[i] = (result_imgs[i] - min_v)/(max_v - min_v)
        # result_img = (result_img - min_v)/(max_v - min_v)
        
        error_maps = [np.abs(gt_img - result_imgs[i]) for i in range(len(result_imgs))]
        std_map = np.std(result_imgs, axis=0, ddof=1)
        mean_prediction = np.mean(result_imgs, axis=0)
        L2 = [(error_maps[i] ** 2).mean() for i in range(len(result_imgs))]
        std_L2 = np.std(L2)
        mean_L2 = np.mean(L2)
        PSNR = [10 * np.log10(1 / L2[i]) for i in range(len(result_imgs)) ]
        std_PSNR = np.std(PSNR)
        mean_PSNR = np.mean(PSNR)
        SSIM = [ssim(gt_img, result_imgs[i]) for i in range(len(result_imgs))]
        std_SSIM = np.std(SSIM)
        mean_SSIM = np.mean(SSIM)
        
        NLL = np.mean(np.log(std_map) + ((mean_prediction-gt_img)**2) / (std_map**2))
        MACE = np.mean(np.abs(std_map -np.abs(gt_img - mean_prediction)))
        
        
        # NLL = [np.mean(-np.log(result_imgs[i])) for i in range(len(result_imgs))]
        # MACE = [np.mean(np.abs(gt_img - result_imgs[i])) for i in range(len(result_imgs))]
        
        # # down sample gt and res img using avg pooling
        # gt_img = torch.tensor(gt_img).unsqueeze(0).unsqueeze(0)
        # result_img = torch.tensor(result_img).unsqueeze(0).unsqueeze(0)
        # # gt_img = torch.nn.functional.avg_pool3d(gt_img, 2)
        # # result_img = torch.nn.functional.avg_pool3d(result_img, 2)
        # gt_img = gt_img.squeeze().squeeze().numpy()
        # result_img = result_img.squeeze().squeeze().numpy()
        # # flatten the image
        # gt_img = gt_img.flatten()
        # result_img = result_img.flatten()

        return mean_PSNR, mean_SSIM, mean_L2, std_L2, std_PSNR, std_SSIM, NLL, MACE, gt_img, result_imgs


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics norm between NII.GZ files.')
    # parser.add_argument('--gt_dir', type=str, default = '/home/shirui/INPAINT/data/augmented_data')
    # parser.add_argument('--mask', type=str, default = '')
    # parser.add_argument('--target', type=str, default = 'flair')

    parser.add_argument('--res', type=str, required=True, help='Directory containing result NII.GZ files')
    parser.add_argument('--batch_size', type=int, default = 1)
    parser.add_argument('--num_workers', type=int, default = 8)
    
    args = parser.parse_args()

    dataset = ResDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    ssim_values = []
    l2_values = []
    psnr_values = []
    ssim_std = []
    l2_std = []
    psnr_std = []
    NLL_values = []
    MACE_values = []
    
    ratio = 1
    gt_imgs = np.zeros((len(dataset), 192*192*152//(ratio**3) ))
    res_imgs = np.zeros((len(dataset), 192*192*152//(ratio**3) ))
    for j, batch in enumerate(tqdm(dataloader)):
        PSNR, SSIM, L2, std_L2, std_PSNR, std_SSIM, NLL, MACE, gt_img, res_img = batch
        # SSIM, PSNR, L2, gt_img, res_img = batch
        # for i in range(len(SSIM)):
        #     viz_plot(gt_img[i], path = "debug/"+str(j*args.batch_size + i))
        for i in range(len(SSIM)):
            ssim_values.append(SSIM[i].item())
            l2_values.append(L2[i].item())
            psnr_values.append(PSNR[i].item())
            ssim_std.append(std_SSIM[i].item())
            l2_std.append(std_L2[i].item())
            psnr_std.append(std_PSNR[i].item())
            NLL_values.append(NLL[i].item())
            MACE_values.append(MACE[i].item())
            
            # gt_imgs[j*args.batch_size + i] = gt_img[i]
            # res_imgs[j*args.batch_size + i] = res_img[i]

    average_ssim = np.mean(ssim_values)
    average_l2 = np.mean(l2_values)
    average_psnr = np.mean(psnr_values)
    std_L2 = np.mean(l2_std)
    std_PSNR = np.mean(psnr_std)
    std_ssim = np.mean(ssim_std)
    NLL = np.mean(NLL_values)
    MACE = np.mean(MACE_values)
    
    # std_ssim = np.std(ssim_values)
    # std_l2 = np.std(l2_values)
    # std_psnr = np.std(psnr_values)
    # print("==============PSNR==============")
    # print(psnr_values)
    # print("==============SSIM==============")
    # print(ssim_values)
    
    print("directory: ", args.res)
    print(f"Number of samples: {len(ssim_values)}")
    print(f"Average SSIM: {average_ssim}")
    print(f"Average L2: {average_l2}")
    print(f"Average PSNR: {average_psnr}")
    print(f"Std SSIM: {std_ssim}")
    print(f"Std L2: {std_L2}")
    print(f"Std PSNR: {std_PSNR}")
    print(f"Average NLL: {NLL}")
    print(f"Average MACE: {MACE}")    
    
    # MMD
    # mmd = mmd_linear(gt_imgs, res_imgs)
    # print(f"mmd_linear: {mmd}")
    # mmd = mmd_rbf(gt_imgs, res_imgs)
    # print(f"mmd_rbf: {mmd}")
    # mmd = mmd_poly(gt_imgs, res_imgs)
    # print(f"mmd_poly: {mmd}")

if __name__ == '__main__':
    main()




# def make_video(data, path):
#     # expected shape of data: 155 192 192
#     frames = scale_data(data)
#     f,w,h = frames.shape
#     frames[0][0][0] = 0
#     frames[-1][-1][-1] = 255

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(path + ".avi", fourcc, 20.0, (w, h),isColor=False)

#     for frame in frames:
#         out.write(frame)

#     # Release everything when job is finished
#     out.release()