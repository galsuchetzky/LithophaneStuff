import lithophane as li
import matplotlib.pylab as plt
import matplotlib.image as img
import numpy as np

from utils import get_channels_CMYK, get_channels_CMY


def read_image(image_path):
    im = img.imread(image_path)

    # Display image.
    plt.imshow(im)
    plt.show()

    return im


def image2grayscale(im):
    gray = li.rgb2gray(im)

    # Display grayscale image.
    plt.imshow(gray, cmap='gray')
    plt.colorbar()
    plt.show()

    return gray


def calcXYZ(gray, h, d):
    # Generate x,y and z values for each pixel
    width = 102  # Width in mm
    x, y, z = li.jpg2stl(gray, width=width, h=h, d=d, show=False)

    # Display.
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.axis('off')
    plt.title('x distances (mm)')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(y)
    plt.axis('off')
    plt.title('y distances (mm)')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(z)
    plt.axis('off')
    plt.title('z distances (mm)')
    plt.colorbar()

    # plot a cross-section to try and visualize data
    plt.plot(x[550, 0:100], z[550, 0:100])
    plt.axis('equal')

    plt.ylabel('Lithophane Depth (z direction mm)')
    plt.title('Lithophane cross-seciton')

    plt.show()

    return x, y, z


def make_flat_model(x, y, z, added_name=''):
    model = li.makemesh(x, y, z)

    # Save model.
    filename = imagefile[:-5].split('/')
    filename[-1] += added_name + '_Flat.stl'
    filename[-2] = 'Output'
    filename = '/'.join(filename)
    model.save(filename)
    print(filename)

    # Display stl. note z axis is not same scale as x and y axes.
    # li.showstl(x, y, z)


def make_cylinder_model(x, y, z):
    cx, cy, cz = li.makeCylinder(x, y, z)
    # li.showstl(cx, cy, cz)

    model = li.makemesh(cx, cy, cz)

    filename = imagefile[:-5].split('/')
    filename[-1] += '_Cylinder.stl'
    filename[-2] = 'Output'
    filename = './'+'/'.join(filename)
    model.save(filename)


def make_model_cmyk(image_path):
    # Read image and split.
    im = img.imread(image_path)
    Y, M, C, K = get_channels_CMYK(im)

    # Show channels
    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    for i, ax, title in zip([Y, M, C, K], axs, ['Y', 'M', 'C', 'K']):
        ax.imshow(i, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
    plt.show()

    for image, name in zip([Y, M, C, K], ['_CMYK_Y', '_CMYK_M', '_CMYK_C', '_CMYK_K']):
        x, y, z = calcXYZ(image, h=1, d=0.1)
        make_flat_model(x, y, z, name)


def make_model_cmy(image_path):
    # Read image and split.
    im = img.imread(image_path)
    Y, M, C = get_channels_CMY(im)

    # Show channels
    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    for i, ax, title in zip([Y, M, C], axs, ['Y', 'M', 'C']):
        ax.imshow(i, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
    plt.show()

    for image, name in zip([Y, M, C], ['_CMY_Y', '_CMY_M', '_CMY_C']):
        x, y, z = calcXYZ(image, h=1, d=0.1)
        make_flat_model(x, y, z, name)


def make_model_grayscale(image_path):
    # Read the image.
    image = read_image(image_path)

    # Convert to Gray Scale.
    gray = image2grayscale(image)

    # Get x, y, z.
    x, y, z = calcXYZ(gray, h=2, d=0.2)

    # Build the flat model.
    make_flat_model(x, y, z)

    # Build the cylinder model.
    # make_cylinder_model(x, y, z)


if __name__ == '__main__':
    imagefile = './src_images/karina.jpg'
    # make_model_cmy(imagefile)
    make_model_grayscale(imagefile)
