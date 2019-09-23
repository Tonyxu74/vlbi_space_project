import numpy as np
import glob
import os


def findFile(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


def findstats(path, endswith):
    folders = glob.glob('{}/*/'.format(path))
    path_list = []

    for imfolder in folders:
        path_list.extend(sorted(findFile(imfolder, endswith)))

    datatype = endswith.replace('.npy', '')

    image_list = []
    for image_path in path_list:
        image_list.append(np.load(image_path))

    image_list = np.asarray(image_list)

    mean = np.mean(image_list)
    std_dev = np.std(image_list)

    print('the current path is: {}'.format(path))
    print('The mean of datatype "{}" is: {}'.format(datatype, mean))
    print('The standard deviation of datatype "{}" is: {}'.format(datatype, std_dev))

    return mean, std_dev


# for the cal256 dataset, which only has amp and phase images
findstats('../data/arrays/train', 'amp_img.npy')
findstats('../data/arrays/train', 'phase_img.npy')

# for the vlbi online dataset, which has amp and phase images and labels
findstats('../data/arrays/val', 'amp_img.npy')
findstats('../data/arrays/val', 'amp_label.npy')
findstats('../data/arrays/val', 'phase_img.npy')
findstats('../data/arrays/val', 'phase_gt.npy')

