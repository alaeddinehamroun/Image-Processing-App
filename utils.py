import numpy as np
import math 
import matplotlib.pyplot as plt
import random
from numpy import unravel_index
def read_pgm(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
    # Reading header
    format = lines[0]
    comment = lines[1]
    n_cols, n_raws = [int(s) for s in lines[2].split() if s.isdigit()]
    p_max = lines[3]

    # Reading data
    data = []
    for line in lines[4:]:
      #print(line)
      data.extend([int(c) for c in line.split()])
      #print([int(c) for c in line.split()])
  data = np.array(data)
  data = data.reshape((n_raws, n_cols))
  return data, format, comment, n_cols, n_raws, p_max
def write_pgm(data, format, comment, n_cols, n_raws, p_max, filename='file'):


  fout=open(filename, 'wb')
  pgm_header = f'{format}{comment}{n_cols} {n_raws}\n{p_max}'
  fout.write(bytearray(pgm_header, 'ascii'))
  for l in data:
    line = l.astype(str).tolist()
    line = ' '.join(line)
    line = line + '\n'
    #print(line)
    fout.write(bytearray(line, 'ascii'))  
  fout.close()
def read_ppm(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
    # Reading header
    format = lines[0]
    comment = lines[1]
    n_cols, n_raws = [int(s) for s in lines[2].split() if s.isdigit()]
    p_max = lines[2]
    # Reading data
    data = []
    for line in lines[4:]:
      #print(line)
      data.extend([int(c) for c in line.split()])
      #print([int(c) for c in line.split()])
  data = np.array(data)
  data = data.reshape((n_raws, n_cols, 3))
  return data, format, comment, n_cols, n_raws, p_max
def apply_filter_rgb(data, filtername, kernel_size=3):
    rChannel = data[:, :, 0]
    gChannel = data[:, :, 1]
    bChannel = data[:, :, 2]
    rFiltered = apply_filter(rChannel, filtername, kernel_size)
    gFiltered = apply_filter(gChannel, filtername, kernel_size)
    bFiltered = apply_filter(bChannel, filtername, kernel_size)
    result = (np.dstack((rFiltered, gFiltered, bFiltered)) * 255.999).astype(np.uint8)
    return result
def apply_filter(data, filtername, kernel_size=3, padding=1, strides=1):
  xKernShape = kernel_size
  yKernShape = kernel_size
  if filtername == "Mean":
    kernel = np.full((int(kernel_size), int(kernel_size)), 1)
  elif filtername == "Gaussian":
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
  imagePadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
  imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
  xOutput = int(((data.shape[0] - xKernShape + 2 * padding) / strides) + 1)
  yOutput = int(((data.shape[1] - yKernShape + 2 * padding) / strides) + 1)
  output = np.zeros((xOutput, yOutput))
  for y in range(data.shape[1]):
    if y > data.shape[1] - yKernShape: 
      break
    if y % strides == 0:
      for x in range(data.shape[0]):
        if x > data.shape[0] - xKernShape:
          break
        try:
          if x % strides == 0:
            if filtername == "Median":
              output[x, y] = np.median(imagePadded[x: x + xKernShape, y: y + yKernShape])
            elif filtername == "Mean":
              output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()/9
            elif filtername == "Gaussian":
              output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()/16

            
        except:
          break

  return output


def img_ecart_type(data):
  e = 0
  nb_pixels = 1
  for i in range(len(data.shape)):
    nb_pixels = nb_pixels*data.shape[i]
  mean = img_mean(data)
  data = data.reshape(nb_pixels)
  for p in data:
    e += (p - mean)**2
  return math.sqrt(e / nb_pixels)
def img_mean(data):
  sum = 0
  nb_pixels = 1
  for i in range(len(data.shape)):
    nb_pixels = nb_pixels*data.shape[i]
  data = data.reshape(nb_pixels)
  for p in data:
    sum += p
  return sum / nb_pixels

def prepare_data_for_hist(data):
  nb_pixels = 1
  for i in range(len(data.shape)):
    nb_pixels = nb_pixels*data.shape[i]

  data_array = data.reshape(nb_pixels)
  H = np.zeros(256, dtype=int)
  for p in data_array:
    H[p] += 1
  count_arr =  np.bincount(data.reshape(nb_pixels))

  return H, count_arr

def create_grayscale_hist(data):
  H, count_arr = prepare_data_for_hist(data)
  plt.plot(np.arange(256), count_arr)
  plt.title("Grayscale Histogram")
  plt.xlabel("grayscale value")
  plt.ylabel("pixel count")
  plt.show()



def create_cumulative_hist(data):
  H, count_arr = prepare_data_for_hist(data)
  cH = []
  cH.append(H[0])
  s = cH[0]
  for p in H[1:]:
    s += p
    cH.append(s)
  carr = np.cumsum(count_arr)
  plt.plot(np.arange(256), carr)
  plt.title("Cumulative Histogram")
  plt.xlabel("grayscale value")
  plt.ylabel("pixel count")
  plt.show()


def signal_to_noise_ratio(data, filter_data):
    S, B = (0, 0)
    mean = img_mean(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            S += (data[i][j] - mean)**2
            B += (filter_data[i][j] - data[i][j])**2
    return math.sqrt(S/B)

def seuillage_indep(data, sR, sG, sB):
    rChannel = data[:, :, 0]
    gChannel = data[:, :, 1]
    bChannel = data[:, :, 2]
   
    for channel, seuil in zip([rChannel, gChannel, bChannel], [sR, sG, sB]):
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if channel[i][j] > seuil:
                    channel[i][j] = 255
                else:
                    channel[i][j] = 0

    return (np.dstack((rChannel, gChannel, bChannel)) * 255.999).astype(np.uint8)
def seuillage_ET(data, sR, sG, sB):
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j][0]> sR and  data[i][j][1]> sG and  data[i][j][2]> sB:
                continue
            else:
                data[i][j][0] = 0
                data[i][j][1] = 0
                data[i][j][2] = 0

    return data
def seuillage_OU(data, sR, sG, sB):
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j][0]>sR or data[i][j][1]>sG or data[i][j][2]>sB:
                continue
            else:
                data[i][j][0] = 0
                data[i][j][1] = 0
                data[i][j][2] = 0

    return data

def threshold_otsu_impl(image, nbins=0.1):
    
    #validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return
    
    #validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, np.max(image)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
            print("trace:", within_class_variance, color_threshold)
            
    return least_variance_threshold

def seuillage_auto_indep(data):
    rChannel = data[:, :, 0]
    gChannel = data[:, :, 1]
    bChannel = data[:, :, 2]
    sR = threshold_otsu_impl(rChannel)
    sG = threshold_otsu_impl(gChannel)
    sB = threshold_otsu_impl(bChannel)
    return seuillage_indep(data, sR, sG, sB), sR, sG, sB

def seuillage_autp_ET(data):
    rChannel = data[:, :, 0]
    gChannel = data[:, :, 1]
    bChannel = data[:, :, 2]
    sR = threshold_otsu_impl(rChannel)
    sG = threshold_otsu_impl(gChannel)
    sB = threshold_otsu_impl(bChannel)
    return seuillage_ET(data, sR, sG, sB), sR, sG, sB

def seuillage_auto_OU(data):
    rChannel = data[:, :, 0]
    gChannel = data[:, :, 1]
    bChannel = data[:, :, 2]
    sR = threshold_otsu_impl(rChannel)
    sG = threshold_otsu_impl(gChannel)
    sB = threshold_otsu_impl(bChannel)

    return seuillage_OU(data, sR, sG, sB), sR, sG, sB
# def enhance_contour():

def apply_noise(data):
  original_shape = data.shape
  nb_pixels = 1
  for i in range(len(data.shape)):
    nb_pixels = nb_pixels*data.shape[i]
  data = data.reshape(nb_pixels)
  for i in range(nb_pixels):
    
    r = random.randint(0, 20)
    if r == 0:
      data[i] = 0
    if r == 20:
      data[i] = 255

  return data.reshape(original_shape)
  




def dilation(data, size):
    if (size % 2 == 0):
        raise Exception('size must be odd')

    padding = 1
    xKernShape = size
    yKernShape = size
    strides = 1
    imagePadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    xOutput = int(((data.shape[0] - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((data.shape[1] - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))
    output = output.astype(np.uint8)
    for y in range(data.shape[1]):
        if y > data.shape[1] - yKernShape: 
            break
        if y % strides == 0:
            for x in range(data.shape[0]):
                if x > data.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        maxindex = imagePadded[x: x + xKernShape, y: y + yKernShape].argmax()
                        dx, dy = unravel_index(maxindex, imagePadded[x: x + xKernShape, y: y + yKernShape].shape)
                        output[x, y] = imagePadded[x+xKernShape+dx, y+yKernShape+dy]
                       
                                    
            
                except:
                    break

    
    return output

def erosion(data, size=3):
    if (size % 2 == 0):
        raise Exception('size must be odd')

    padding = 1
    xKernShape = size
    yKernShape = size
    strides = 1
    imagePadded = np.zeros((data.shape[0] + padding*2, data.shape[1] + padding*2))
    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = data
    xOutput = int(((data.shape[0] - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((data.shape[1] - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))
    output = output.astype(np.uint8)
    for y in range(data.shape[1]):
        if y > data.shape[1] - yKernShape: 
            break
        if y % strides == 0:
            for x in range(data.shape[0]):
                if x > data.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        minindex = imagePadded[x: x + xKernShape, y: y + yKernShape].argmin()
                        dx, dy = unravel_index(minindex, imagePadded[x: x + xKernShape, y: y + yKernShape].shape)
                        output[x, y] = imagePadded[x+xKernShape+dx, y+yKernShape+dy]
                       
                                    
            
                except:
                    break
    return output

def opening(data, size):
    d = erosion(data, size)
    return dilation(d, size)

def closing(data, size):
    d = dilation(data, size)
    return erosion(d, size)

def binarize(data, threshold):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] < threshold:
                data[i][j] = 0
            else:
                data[i][j] = 255
    return data
# rChannel = data[:, :, 0]
#     gChannel = data[:, :, 1]
#     bChannel = data[:, :, 2]
#     rFiltered = apply_filter(rChannel, filtername)
#     gFiltered = apply_filter(gChannel, filtername)
#     bFiltered = apply_filter(bChannel, filtername)
#     result = (np.dstack((rFiltered, gFiltered, bFiltered)) * 255.999).astype(np.uint8)
#     return result

def rgb2gray(data):
    g_data = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            g_data[i][j] = data[i][j][0]*0.299 + data[i][j][1]*0.587 + data[i][j][2]*0.144
    return g_data.astype(np.uint8)