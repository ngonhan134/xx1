# some useful functions...
from functools import reduce
from unicodedata import name
from sklearn.decomposition import KernelPCA
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import cv2

im_side = 64


def derivate_image(im, angle):
    h, w = np.shape(im)
    pad_im = np.pad(im, (1, 0), 'edge')
    if angle == 'horizontal':  # horizontal derivative
        deriv_im = pad_im[1:, :w] - im  # [1:, :w]
    elif angle == 'vertical':
        deriv_im = pad_im[:h, 1:] - im  # [1:, :w]

    return deriv_im

def extract_ltrp1(im_d_x, im_d_y):
    encoded_image = np.zeros(np.shape(im_d_y))  # define empty matrix, of the same shape as the image...

    # # apply conditions for each orientation...
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y >= 0)] = 1
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y >= 0)] = 2
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y < 0)] = 3
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y < 0)] = 4

    return encoded_image
def extract_ltrp2(ltrp1_code, plotting_flag=False):

    this_im_side = np.shape(ltrp1_code)[0]
    ltrp1_code = np.pad(ltrp1_code, (1, 1), 'constant', constant_values=0)
    g_c1 = np.zeros((3, this_im_side, this_im_side))
    g_c2 = np.zeros((3, this_im_side, this_im_side))
    g_c3 = np.zeros((3, this_im_side, this_im_side))
    g_c4 = np.zeros((3, this_im_side, this_im_side))

    for i in range(1, im_side+1):
        for j in range(1, im_side+1):
            g_c = ltrp1_code[i, j]

            # # extract neighborhood around g_c pixel
            neighborhood = np.array([ltrp1_code[i + 1, j], ltrp1_code[i + 1, j - 11], ltrp1_code[i, j - 11],
                                     ltrp1_code[i - 1, j - 1], ltrp1_code[i - 1, j], ltrp1_code[i - 1, j + 1],
                                     ltrp1_code[i, j + 1], ltrp1_code[i + 1, j + 1]])
            # # determine the codes that are different from g_c
            mask = neighborhood != g_c
            # # apply mask
            ltrp2_local = np.multiply(neighborhood, mask)

            # # construct P-components for every orientation.
            if g_c == 1:
                for direction_index, direction in enumerate([2, 3, 4]):
                    g_dir = ltrp2_local == direction
                    g_c1[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=int))

            elif g_c == 2:
                for direction_index, direction in enumerate([1, 3, 4]):
                    g_dir = ltrp2_local == direction
                    g_c2[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=int))

            elif g_c == 3:
                for direction_index, direction in enumerate([1, 2, 4]):
                    g_dir = ltrp2_local == direction
                    g_c3[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=int))

            elif g_c == 4:
                for direction_index, direction in enumerate([1, 2, 3]):
                    g_dir = ltrp2_local == direction

                    g_c4[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=int))
                    pass

            elif g_c not in [1, 2, 3, 4]:
                raise Exception('Error - Invalid value for g_c. List of possible values include [1,2,3,4].')

    large_g_c = []
    for this_g_c in [g_c1, g_c2, g_c3, g_c4]:
        large_g_c.extend(this_g_c)
    large_g_c = np.array(large_g_c)

    return large_g_c

def extract_compcode_with_magnitude(input_image, no_theta=6, sigma=1.5):

    theta = np.arange(1, no_theta + 1) * np.pi / no_theta
    (x, y) = np.meshgrid(np.arange(0, 35, 1), np.arange(0, 35, 1))
    xo, yo = np.shape(x)[0] / 2, np.shape(x)[0] / 2
    kappa = np.sqrt(2. * np.log(2.)) * ((np.power(2, sigma) + 1.) / ((np.power(2, sigma) - 1.)))
    omega = kappa / sigma

    Psi = {}  # where the filters are stored
    gabor_responses = []
    for i in range(0, len(theta)):
        xp = (x - xo) * np.cos(theta[i]) + (y - yo) * np.sin(theta[i])
        yp = -(x - xo) * np.sin(theta[i]) + (y - yo) * np.cos(theta[i])
        # Directional Gabor Filter
        Psi[str(i)] = (-omega / (np.sqrt(2 * np.pi)) * kappa) * \
                      np.exp(
                          (-np.power(omega, 2) / (8 * np.power(kappa, 2))) * (4 * np.power(xp, 2) + np.power(yp, 2))) * \
                      (np.cos(omega * xp) - np.exp(-np.power(kappa, 2) / 2))

        # # used for debugging... #1
        # plt.subplot(2,3,i+1)
        # plt.imshow(Psi[str(i)], cmap='jet')
        filtered = convolve2d(input_image, Psi[str(i)], mode='same', boundary='symm')
        # # used for debugging #2
        # plt.imshow(filtered)
        # plt.show()
        gabor_responses.append(filtered)

    # plt.show() #1
    gabor_responses = np.array(gabor_responses)

    compcode_orientations = np.argmin(gabor_responses, axis=0)
    compcode_magnitude = np.min(gabor_responses, axis=0)

    return compcode_orientations, compcode_magnitude


def derivate_image_palm_line(im, angle, m1=3, m2=1, N1=3, N2=4):

    pad_im = np.pad(im, (m2, m1), 'edge')  # image needs to be padded to the right and bellow.
    h, w = np.shape(pad_im)

    deriv_im = np.zeros(np.shape(im))

    if angle == 'horizontal':
        # # moving window across the image...
        for i in range(m2, h - m1):
            for j in range(m2, w - m1):
                g_c = pad_im[i, j]  # current g_c
                e1_sum = 0  # used for first sum in the equation
                for k in range(0, m1):
                    e1_sum += pad_im[i, j + k]
                element1 = (e1_sum + g_c) / float((m1 + 1))  # initially had N1 in the last paranthesis.
                # efficiently replaced with (m1+1)

                e2_sum = 0  # used for second sum in the equation
                for k in range(0, m2):
                    e2_sum += (pad_im[i, j + k] + pad_im[i - k, j] + pad_im[i - k, j] + pad_im[
                        i + k, j])  # (pad_im[i,j+k] + pad_im[i,j-k])
                element2 = e2_sum / float(m2 * 4)  # initially had N2 in the last paranthesis.
                # efficiently replaced with (m2*4)

                # # used for debugging
                # res = element1 - element2

                deriv_im[i - m2, j - m2] = element1 - element2

    elif angle == 'vertical':
        # # moving window across the image...
        for i in range(m2, h - m1):
            for j in range(m2, w - m1):
                g_c = pad_im[i, j]  # current g_c
                e1_sum = 0  # used for first sum in the equation
                for k in range(0, m1):
                    e1_sum += pad_im[i + k, j]
                    # + (pad_im[i, j + k] + pad_im[i - k, j] + pad_im[i - k, j] + pad_im[i + k, j])

                element1 = (e1_sum + g_c) / float((m1 + 1))  # initially had N1 in the last paranthesis.
                # efficiently replaced with (m1+1)

                e2_sum = 0  # used for second sum in the equation
                for k in range(0, m2):
                    e2_sum += (pad_im[i, j + k] + pad_im[i - k, j] + pad_im[i - k, j] + pad_im[
                        i + k, j])  # + (pad_im[i+k, j] + pad_im[i+k, j])
                element2 = e2_sum / float(m2 * 4)  # initially had N2 in the last paranthesis.
                # efficiently replaced with (m2*4)

                # # used for debugging
                # res = element1 - element2
                deriv_im[i - m2, j - m2] = element1 - element2

    return deriv_im

def extract_ltrp2_hist(ltrp2_code, block_size=8, no_bins=8, hist_range=[0, 255]):

    hist_ltrp2 = []  # container for all P-component
    n_blocks = (im_side // block_size)
    for P_index, P_component in enumerate(ltrp2_code):
        # for every p component in the computed LTrP2 feature...

        p_index_feature = np.zeros((n_blocks, n_blocks * block_size))  # empty P-component container...
        block_counter = 0  # keeping track of processed blocks (within the row)
        previous_i = 0  # starting position for i, used for extracting values (updated every after every row)
        row_counter = 0  # keeping track of processed rows...

        for i in np.arange(block_size, im_side + block_size, block_size):
            row = np.array([])  # where histogram values are stored...
            previous_j = 0  # starting position for j, used for extracting values (updated every after every block)

            for j in np.arange(block_size, im_side + block_size, block_size):
                block = P_component[previous_i:i, previous_j:j]
                block_hist = cv2.calcHist([np.uint8(block)], channels=[0], mask=None,
                                          histSize=[no_bins], ranges=hist_range)

                # # used for debugging...
                # plt.plot(a_hist/np.sum(a_hist),marker='o')
                # plt.show()

                # store block histogram in row... but normalize values before!
                row = np.concatenate((row, np.reshape(np.array(block_hist / np.sum(block_hist)), (no_bins))), axis=-1)

                block_counter += 1
                previous_j = j

            p_index_feature[row_counter, :] = np.array(row)

            row_counter += 1
            previous_i = i

            del row

        p_index_feature = np.array(p_index_feature)
        hist_ltrp2.append(p_index_feature)

    # clear some memory...
    del block, block_hist, p_index_feature, P_component, ltrp2_code,
    final_h_ltrp2 = np.zeros((np.shape(hist_ltrp2)[1] *  # number of rows in each p_feature... (n_blocks)
                              np.shape(hist_ltrp2)[0],  # number of p_features (12)
                              np.shape(hist_ltrp2)[2]))  # number of bins in each histogram * n_blocks

    for hist_ltrp2_index, this_hist_ltrp2 in enumerate(hist_ltrp2):
        final_h_ltrp2[hist_ltrp2_index * np.shape(hist_ltrp2)[1]:
        (hist_ltrp2_index + 1) * np.shape(hist_ltrp2)[1], :] = this_hist_ltrp2

    return final_h_ltrp2

pca1 = KernelPCA(n_components=128, kernel='linear')
pca2 = KernelPCA(n_components=15, kernel='linear')


def extract_local_tetra_pattern_palm(image,
                                     input_mode='grayscale', theta_orientations=12, comp_sigma=1.5,
                                     derivative_mode='palmprint', m1=3, m2=1, plot_figures_flag=False,
                                     block_size=8, n_bins=8, h_range=[0, 255],
                                     pca_no_components=15):
    if input_mode == 'grayscale':
        pass
    elif input_mode == 'gabor':
        # # Obtain the Gabor-filter response (using min rule) for input image.
        orientations, image = extract_compcode_with_magnitude(image, no_theta=theta_orientations, sigma=comp_sigma)
        # # normalize values
        image = (image - np.max(np.max(image))) * -1
        image = (image / np.max(np.max(image))) * 255
    else:
        raise 'Unknown value for "input_mode". Either "grayscale" or "gabor" are accepted.'
    #####################################################
    # # Compute the Derivative on the horizontal and vertical...
    #####################################################
    if derivative_mode == 'standard':
        deriv_h = derivate_image(image, 'horizontal')
        deriv_v = derivate_image(image, 'vertical')

    elif derivative_mode == 'palmprint':
        # # As was defined for Palmpritn recognition...
        deriv_h = derivate_image_palm_line(im=image, angle='horizontal', m1=m1, m2=m2)
        deriv_v = derivate_image_palm_line(im=image, angle='vertical', m1=m1, m2=m2)
    else:
        raise 'Unknown value for "derivative_mode". Either "standard" or "palmprint" are accepted.'
    ######################################################
    # # Extract LTrP1 code...
    ######################################################
    ltrp1 = extract_ltrp1(im_d_x=deriv_h, im_d_y=deriv_v)
    ######################################################
    # # Plot, if flag is True...
    ######################################################
  

    ######################################################
    # # Extract LTrP2 P-components, based on the previously obtained LTrP1
    ######################################################

    ltrp2 = extract_ltrp2(ltrp1, plotting_flag=plot_figures_flag)

    ######################################################
    # # Extract histogram feature of LTrP2
    ######################################################
    ltrp2_hist = extract_ltrp2_hist(ltrp2_code=ltrp2, block_size=block_size, no_bins=n_bins, hist_range=h_range)

    ######################################################
    # # Decompose the LTrP2-histogram feature into fewer components...
    ######################################################
    pca = KernelPCA(n_components=pca_no_components, kernel='linear')
    decomposed_feature = pca.fit_transform(X=ltrp2_hist)
    
    return decomposed_feature


def LMTRP_process(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (im_side, im_side))
    image = np.array(image, dtype=np.float32)  # /255.

    compcode_orientations, compcode_magnitude = extract_compcode_with_magnitude(image)

    compcode_magnitude = (compcode_magnitude - np.max(np.max(compcode_magnitude))) * -1
    compcode_magnitude = (compcode_magnitude / np.max(np.max(compcode_magnitude))) * 255

    deriv_h = derivate_image(im=compcode_magnitude, angle='horizontal')
    deriv_v = derivate_image(im=compcode_magnitude, angle='vertical')

    deriv_h = derivate_image_palm_line(im=compcode_magnitude, angle='horizontal', m1=3, m2=1)
    deriv_v = derivate_image_palm_line(im=compcode_magnitude, angle='vertical', m1=3, m2=1)

    ltrp1 = extract_ltrp1(im_d_x=deriv_h, im_d_y=deriv_v)

    plot_figures_flag=False
    ltrp2 = extract_ltrp2(ltrp1, plotting_flag=plot_figures_flag)

    ltrp2_hist = extract_ltrp2_hist(ltrp2_code=ltrp2, block_size=8, no_bins=8, hist_range=[0,255])

    final_lmtrp_feature = pca2.fit_transform(X=ltrp2_hist)

    final_lmtrp_feature = extract_local_tetra_pattern_palm(image,
                                                 input_mode='gabor',theta_orientations=12,
                                                 comp_sigma=1.5,
                                                 derivative_mode='palmprint', m1=3,m2=1,
                                                 plot_figures_flag=False,
                                                 block_size=8,n_bins=8)
    return final_lmtrp_feature
