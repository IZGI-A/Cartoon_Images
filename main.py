from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage

def show_image(title, image):
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

""" PART 1 """
def image_smoothing(image, filter_type, filter_parameter_value):
    image_array = np.array(image)
    if filter_type == 'gaussian':
        smoothed_image = gaussian_filter(image_array, filter_parameter_value)

    elif filter_type == 'median':
        smoothed_image = median_filter(image_array, size=filter_parameter_value)

    elif filter_type == 'uniform':
        smoothed_image = ndimage.uniform_filter(image_array, size=filter_parameter_value)

    smoothed_image = Image.fromarray(np.uint8(smoothed_image))
    return smoothed_image


""" PART 2 """
def edge_detection(image, k, sigma, epsilon):
    sigma2 = k * sigma
    gray_image = image.convert('L')
    input_image = np.array(gray_image)
    gaussian1 = gaussian_filter(input_image, sigma=sigma)
    gaussian2 = gaussian_filter(input_image, sigma=sigma2)

    equation1and2_result = gaussian1 - gaussian2
    thresholded_image = np.where(equation1and2_result >= epsilon, 255, 0).astype(np.uint8)
    thresholded_image = Image.fromarray(np.uint8(thresholded_image))

    return thresholded_image


def image_quantization(image, conversion, value):
    image_rgb = image.convert('RGB') #since original image is RGBA
    converted_image = image_rgb.convert(conversion)
    image_rgb_array = np.array(image_rgb)
    converted_image_array = np.array(converted_image)
    if conversion == 'LAB':
        quantized = np.floor_divide(converted_image_array[:, :, 0], value) * value
        converted_image_array[:, :, 0] = quantized
    elif conversion == 'HSV':
        quantized = np.floor_divide(converted_image_array[:, :, 2], value) * value
        converted_image_array[:, :, 2] = quantized
    else:
        r, g, b = image_rgb.split()
        quantized_r = np.floor_divide(image_rgb_array[:, :, 0], value) * value
        quantized_g = np.floor_divide(image_rgb_array[:, :, 1], value) * value
        quantized_b = np.floor_divide(image_rgb_array[:, :, 2], value) * value
        image_rgb_array[:, :, 0] = quantized_r
        image_rgb_array[:, :, 1] = quantized_g
        image_rgb_array[:, :, 2] = quantized_b

        # quantized_r = r.point(lambda i: i // value * value)
        # quantized_g = g.point(lambda i: i // value * value)
        # quantized_b = b.point(lambda i: i // value * value)
        quantized = Image.fromarray(image_rgb_array)
        return quantized  # Return the quantized image here for 'RGB' conversion

    quantized_image = Image.fromarray(converted_image_array, conversion)
    quantized_image_rgb = quantized_image.convert('RGB')
    return quantized_image_rgb


def combine_edges_quantized(edges, quantized):
    inverted_edges = 1 - edges
    #inverted_edge = Image.fromarray(np.uint8(inverted_edges))
    #inverted_edge.show()
    quantized_array = np.array(quantized)
    combined_image = np.zeros_like(quantized_array)
    for channel in range(quantized_array.shape[-1]):
        print(channel)
        combined_image[:, :, channel] = inverted_edges * quantized_array[:, :, channel]

    combined_image = Image.fromarray(np.uint8(combined_image))
    return combined_image


input_folder = 'report/data/'
output_folder = 'report/result/'

"""------------------ PARAMETERS -----------------------"""

filter_type = 'gaussian' # gaussian, median, uniform
filter_parameter_value = 2 # sigma or kernel_size value
k = 2
sigma_value_for_edge_detection = 2
epsilon = 50
conversion_type = 'LAB' # it can be LAB, HSV, RGB
quantization_value = 1

""" ----------------------------------------------------"""

os.makedirs(output_folder, exist_ok=True)


for i in range(1, 7):
    image_path = f'{input_folder}image{i}.png'
    output_path = f'{output_folder}result{i}.png'


    image = Image.open(image_path)
    #image.show()
    smooth_image = image_smoothing(image, filter_type=filter_type, filter_parameter_value=filter_parameter_value)
    #smooth_image.show()

    edge_detected_image = edge_detection(smooth_image, k=k, sigma=sigma_value_for_edge_detection, epsilon=epsilon)
    #edge_detected_image.show()
    quantized_image = image_quantization(image, conversion=conversion_type, value=quantization_value)
    #quantized_image.show()

    edge_detected_image = np.array(edge_detected_image) / 255
    final_image = combine_edges_quantized(edge_detected_image, quantized_image)
    #final_image.show()
    final_image.save(output_path)
    print(f'{image_path} saved as {output_path}')

    plt.subplots_adjust(wspace=0.03, hspace=0)

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # plt.subplot(1, 4, 2)
    # plt.imshow(smooth_image)
    # plt.title('Smoothed Image')
    # plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(quantized_image, cmap='gray')
    plt.title('Quantized Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(final_image, cmap='gray')
    plt.title('Cartoon Image')
    plt.axis('off')

    plt.show()


    # plt.subplot(2, 3, 1)
    # plt.imshow(image)
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 2)
    # plt.imshow(smooth_image)
    # plt.title('Smoothed Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 3)
    # plt.imshow(edge_detected_image, cmap='gray')
    # plt.title('Edge Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 4)
    # plt.imshow(quantized_image)
    # plt.title('Quantized Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 5)
    # plt.imshow(final_image)
    # plt.title('Cartoon Image')
    # plt.axis('off')

    #plt.tight_layout()
    #plt.show()


