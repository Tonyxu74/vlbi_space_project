import oifits
import os
from astropy.io import fits
from myargs import args
import numpy as np
from tqdm import tqdm

'''
    make some open all files with oifits ending and find their corresponding image, resize to 64x64, 
    and then convert all to fft2 & .png files and save to a folder
    
    also convert every piece of data to an image using UV normalization to 64x64 with each b/w pixel being a vis or 
    vis^2 value
    
    this will be dataset and perhaps automate the whole ordeal to input just the folder, "UVdata", and it'll do it all
    for you that would be great cuz we gonna have a lot of folders fuck 
'''


def find_file(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


def process_datafolder(path):

    oifits_data = find_file(path, '.oifits')
    oifits_data = [oifits_datum for oifits_datum in oifits_data if 'normalized' not in oifits_datum]
    gt_images = find_file(path, '.fits')
    telescope_setting = path.replace('../raw_data/', '').replace('_data', '')

    if not os.path.exists('../data/arrays/{}/'.format(telescope_setting)):
        os.mkdir('../data/arrays/{}/'.format(telescope_setting))

    for data, gt_path in tqdm(zip(oifits_data, gt_images)):
        image_name = gt_path.replace('{}\\targetImgs\\'.format(path), '').replace('.fits', '')
        raw_data = oifits.open(data, quiet=True)
        vcoords = []
        ucoords = []
        visibilities = []
        phases = []
        for visdata in raw_data.vis:
            vcoords.append(visdata.vcoord)
            ucoords.append(visdata.ucoord)
            visibilities.append(visdata._visamp[0])
            phases.append(visdata._visphi[0])

            vcoords.append(-visdata.vcoord)
            ucoords.append(-visdata.ucoord)
            visibilities.append(visdata._visamp[0])
            phases.append(-visdata._visphi[0])

        image = fits.open(gt_path)[0].data
        fft_image = np.fft.fft2(image)
        fft_image = np.fft.fftshift(fft_image)

        image_size = args.imageDims[0]

        amp_gt = np.abs(fft_image)
        phase_gt = np.angle(fft_image, deg=False)

        uvmax = max(max(ucoords), max(vcoords))
        range_ucoords = [int(ucoord * image_size // 2 / uvmax) for ucoord in ucoords]
        range_vcoords = [int(vcoord * image_size // 2 / uvmax) for vcoord in vcoords]

        amp_image = np.zeros(shape=(image_size, image_size))
        phase_image = np.zeros(shape=(image_size, image_size))

        for u, v, v_amp, v_phase in zip(range_ucoords, range_vcoords, visibilities, phases):
            u += image_size // 2 - 1
            v += image_size // 2 - 1

            # u is the x coord, meaning column, which is the second entry in the array, vice versa for v
            amp_image[v][u] = v_amp
            phase_image[v][u] = v_phase

        np.save('../data/arrays/{}/{}_phase_gt.npy'.format(telescope_setting, image_name), phase_gt)
        np.save('../data/arrays/{}/{}_amp_gt.npy'.format(telescope_setting, image_name), amp_gt)
        np.save('../data/arrays/{}/{}_phase_img.npy'.format(telescope_setting, image_name), phase_image)
        np.save('../data/arrays/{}/{}_amp_img.npy'.format(telescope_setting, image_name), amp_image)


if __name__ == "__main__":
    process_datafolder('../raw_data/SgrA-star_data')
    process_datafolder('../raw_data/3C273_data')
    process_datafolder('../raw_data/3C279_data')
    process_datafolder('../raw_data/M87_data')
    process_datafolder('../raw_data/size_data')
    process_datafolder('../raw_data/loc_data')
    process_datafolder('../raw_data/challenge_data')
