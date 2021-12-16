def Main(input_leopold = False):

  # Import libs

  from __future__ import print_function
  import matplotlib.pyplot as plt
  %matplotlib inline

  import os
  #os.environ['CUDA_VISIBLE_DEVICES'] = '3'

  import numpy as np
  from models.skip import *

  import torch
  import torch.optim

  from skimage.metrics import peak_signal_noise_ratio
  from utils.denoising_utils import *

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark =True
  dtype = torch.cuda.FloatTensor

  imsize =-1
  PLOT = True
  sigma = 25
  sigma_ = sigma/255.

  if input_leopold:
    from google.colab import files
    from IPython.display import Image
    uploaded = files.upload()
    ## denoising
    fname = 'image.csv'
  else:
    fname = 'data/F16_GT.png'
  #fname = '11b_194.png'

  # Load image

  def get_noisy_image(img_np, sigma):
      """Adds Gaussian noise to an image.

      Args: 
          img_np: image, np.array with values from 0 to 1
          sigma: std of the noise
      """
      img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
      img_noisy_pil = np_to_pil(img_noisy_np)

      return img_noisy_pil, img_noisy_np

  #img = Image.open('11b_194.png')
  #img_np = np.asarray(img)
  #print(img_np)

  if input_leopold:
    #from numpy import genfromtxt
    #img_np = genfromtxt('image.csv', delimiter=',')
    #print(img_np)

    from PIL import Image
    import matplotlib.image as mpimg
    import numpy as np
    from matplotlib import cm
    #img = mpimg.imread("11b_194.png")
    print(img_np)
    #if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    #    img = (img * 255).astype(np.uint8)
    img_np = (img_np * 255).astype(np.uint8)
    im = Image.fromarray(img_np)
    #im

  if fname == 'data/F16_GT.png':
      # Add synthetic noise
      img_pil = crop_image(get_image(fname, imsize)[0], d=32)
      img_np = pil_to_np(img_pil)

      img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
      if PLOT:
          plot_image_grid([img_np, img_noisy_np], 4, 6);

  elif fname == '11b_194.png':
      # Add synthetic noise
      img = mpimg.imread(fname)
      if img.dtype == np.float32:
          img = (img * 255).astype(np.uint8)
      im = Image.fromarray(img)

      img_pil = crop_image(im, d=32)
      ar = np.array(img_pil)[None, ...]
      img_np = ar.astype(np.float32) / 255

      img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
      if PLOT:
          plot_image_grid([img_np, img_noisy_np], 4, 6);

  elif fname == 'image.csv':
      # Add synthetic noise
      img_np = (img_np * 255).astype(np.uint8)
      im = Image.fromarray(img_np)

      img_pil = crop_image(im, d=32)
      ar = np.array(img_pil)[None, ...]
      img_np = ar.astype(np.float32) / 255

      img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
      if PLOT:
          plot_image_grid([img_np, img_noisy_np], 4, 6);

  else:
      assert False

  # Setup

  INPUT = 'noise' # 'meshgrid'
  pad = 'reflection'
  OPT_OVER = 'net' # 'net,input'

  reg_noise_std = 1./30. # set to 1./20. for sigma=50
  LR = 0.01

  OPTIMIZER='adam' # 'LBFGS'
  show_every = 10
  exp_weight=0.99

  if fname == 'data/F16_GT.png':
      num_iter = 3000
      input_depth = 32 
      figsize = 4 


      net = get_net(input_depth, pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear').type(dtype)

  elif fname == 'image.csv':
      num_iter = 1000
      input_depth = 1
      figsize = 5 

      net = skip(
                  input_depth, 1, 
                  num_channels_down = [128, 128, 128, 128, 128], 
                  num_channels_up   = [128, 128, 128, 128, 128],
                  num_channels_skip = [4, 4, 4, 4, 4], 
                  upsample_mode='bilinear',
                  need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

      net = net.type(dtype)

  else:
      assert False

  net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

  # Compute number of parameters
  s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
  print ('Number of params: %d' % s)

  # Loss
  mse = torch.nn.MSELoss().type(dtype)

  img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

  # Optimize

  PLOT = True
  net_input_saved = net_input.detach().clone()
  noise = net_input.detach().clone()
  out_avg = None
  last_net = None
  psrn_noisy_last = 0

  i = 0
  def closure():

      global i, out_avg, psrn_noisy_last, last_net, net_input

      if reg_noise_std > 0:
          net_input = net_input_saved + (noise.normal_() * reg_noise_std)

      out = net(net_input)

      # Smoothing
      if out_avg is None:
          out_avg = out.detach()
      else:
          out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

      total_loss = mse(out, img_noisy_torch)
      total_loss.backward()

      psrn_noisy = peak_signal_noise_ratio(img_noisy_np, out.detach().cpu().numpy()[0]) 
      psrn_gt    = peak_signal_noise_ratio(img_np, out.detach().cpu().numpy()[0]) 
      psrn_gt_sm = peak_signal_noise_ratio(img_np, out_avg.detach().cpu().numpy()[0]) 

      # Note that we do not have GT for the "snail" example
      # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
      print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
      if  PLOT and i % show_every == 0:
          out_np = torch_to_np(out)
          plot_image_grid([np.clip(out_np, 0, 1), 
                           np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
          print("noisy / output : ", psrn_noisy, "initial / output : ", psrn_gt, "initial / avg_output : " , psrn_gt_sm)



      # Backtracking
      if i % show_every:
          if psrn_noisy - psrn_noisy_last < -5: 
              print('Falling back to previous checkpoint.')

              for new_param, net_param in zip(last_net, net.parameters()):
                  net_param.data.copy_(new_param.cuda())

              return total_loss*0
          else:
              last_net = [x.detach().cpu() for x in net.parameters()]
              psrn_noisy_last = psrn_noisy

      i += 1

      return total_loss

  p = get_params(OPT_OVER, net, net_input)
  optimize(OPTIMIZER, p, closure, LR, num_iter)

  # Final render

  out_np = torch_to_np(net(net_input))
  q = plot_image_grid([np.clip(out_np, 0, 1), img_np, img_noisy_np], factor=13);
