import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

# TAKEN FROM
# https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71


def process_image(image_path):
    image = cv2.imread(image_path)
    colour_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return colour_image


def rgb_to_hex_converter(color):
    # receives a tuple with RGB Colors, like this: (0,128,0)
    hex_color = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"  # :02x is the hex color format
    return hex_color


def find_colors(image, number_of_colors, display_color_chart):
    # reduce the time needed to extract the colors from the image
    resized_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)

    # KMeans expects the input to be of two dimensions, so we use Numpy’s reshape function to reshape the image data.
    modified_image = resized_image.reshape(resized_image.shape[0] * resized_image.shape[1], 3)

    # KMeans algorithm creates clusters based on the supplied count of clusters.
    # In our case, it will form clusters of colors and these clusters will be our top N colors
    model = KMeans(n_clusters=number_of_colors)

    # We then fit and predict on the same image to extract the prediction into the variable labels
    labels = model.fit_predict(modified_image)

    # We use Counter to get count of all labels.
    counts = Counter(labels)

    # To find the colors, we use model.cluster_centers_
    center_colors = model.cluster_centers_

    # The ordered_colors iterates over the keys present in count, and then divides each value by 255
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]

    hex_colors = [rgb_to_hex_converter(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if display_color_chart:
        plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
        plt.show()

    return hex_colors, rgb_colors


# # Run Tests - Note: Proportions may not be reliable as KMeans chooses different centroid at random in each different try
# selected_test_image = process_image('write_image_path_here.extension_as_well')
# color_hex_codes, color_rgb_codes = find_colors(image=selected_test_image, number_of_colors=10, display_color_chart=True)

