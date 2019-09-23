import numpy as np
import random
import math
from PIL import Image


def uv_generate(uvnum, telescope_num=10, datapoints=100, rotation=120, output_size=1000):
    '''
    A function that returns a given amount of UV plane simulations with several parameters

    :param telescope_num: The number of telescopes in the simulation
    :param uvnum: the number of UV plane simulations to return
    :param datapoints: the number of datapoints to take for each telescope
    :param rotation: the amount the earth rotates (in degrees)
    :param output_size: the output width/height of the image in pixels
    :return:
    '''
    earthrad = output_size // 2
    delta_theta = rotation / datapoints

    uv_imgs = []
    for uvimage_num in range(uvnum):
        img_plane_phi = random.random() * 90

        uv_coords = []
        for t in range(telescope_num):
            theta = random.random() * 359
            phi = random.random() * 180
            radius = random.random() * earthrad

            for pt_num in range(datapoints):
                # x y z values corresponding to each rotation of the earth for a certain telescope
                x = radius * np.cos(math.radians(theta + pt_num * delta_theta)) * np.sin(math.radians(phi))
                y = radius * np.sin(math.radians(theta + pt_num * delta_theta)) * np.sin(math.radians(phi))
                z = radius * np.cos(math.radians(phi))

                # these same values inside the plane "pi"
                # xpi = x + rsin(phi) - sin(phi)(xsin(phi) + zcos(phi))
                x_pi = x + earthrad * np.sin(math.radians(img_plane_phi)) - np.sin(math.radians(img_plane_phi)) * \
                    (x * np.sin(math.radians(img_plane_phi)) + z * np.cos(math.radians(img_plane_phi)))

                # zpi = z + rcos(phi) - cos(phi)(xsin(phi) + zcos(phi))
                z_pi = z + earthrad * np.cos(math.radians(img_plane_phi)) - np.cos(math.radians(img_plane_phi)) * \
                    (x * np.sin(math.radians(img_plane_phi)) + z * np.cos(math.radians(img_plane_phi)))

                # the distances are calculated from the line where the plane touches the sphere:
                # v = sqrt((rcos(phi) - zpi))^2 + (rsin(phi) - xpi)^2)
                # v is positive if z value is greater than the "pivot point" which is rcos(phi)
                # u is simply the y term: it is not transformed
                v_term = np.sqrt((earthrad * np.cos(math.radians(img_plane_phi)) - z_pi)**2 +
                                 (earthrad * np.sin(math.radians(img_plane_phi)) - x_pi)**2)
                v = v_term if z_pi > earthrad * np.cos(math.radians(img_plane_phi)) else v_term * -1
                u = y

                # every uv term has a corresponding negative, to account for both directions of interferometry:
                neg_u = -u
                neg_v = -v

                # add the radius to make sure array indices are greater than 0
                v += earthrad
                u += earthrad
                neg_u += earthrad
                neg_v += earthrad

                uv_coords.append([u, v])
                uv_coords.append([neg_u, neg_v])

        uv_image = np.zeros(shape=(earthrad * 2, earthrad * 2)).astype(np.uint8)
        uv_coords = np.asarray(uv_coords).astype(np.uint16)

        for uv_coord in uv_coords:
            # append v first as it is the "column" of the image and u is the "row"
            uv_image[uv_coord[1]][uv_coord[0]] = 1

        uv_image = Image.fromarray(uv_image)
        uv_imgs.append(uv_image)

    return uv_imgs


uv_generate(uvnum=10, output_size=256, rotation=120)


