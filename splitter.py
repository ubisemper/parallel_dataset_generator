import os
import shutil
import numpy as np
import sys
sys.path.append('/home/wkraal/led_system_modelling')
import NoiseModule as nm
import SimpaHelper as sh
from src.utils.cyberdyne_led_array_system import CyberdyneLEDArraySystem
from simpa import Tags
import numpy as np
from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
import h5py
import simpa as sp
from PIL import Image
import os
import wandb
import uuid
from multiprocessing import Pool, Manager
import sklearn
import PIL
from sklearn.model_selection import train_test_split
import json

tmpdir = os.environ.get('TMPDIR')

manager = Manager()
image_log = manager.list()

def add_blur(path):
    print(f'### Blur is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()

    kernel_size = np.random.randint(3, 25)
    intensity_blur = np.random.random()
    
    sinogram = simpa_helper.extract_sinogram()
    sino_ver_blur = nm.random_blur(sinogram, kernel_size, intensity_blur)

    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')
    
    # Save sinogram from the original file to a new hdf5 file to work from
    
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
        'path': path,
        'kernel_size': kernel_size,
        'intensity_blur': intensity_blur
    }

    image_log.append(log)

    return data

def add_ghost(path):
    print(f'### Ghost is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    
    sinogram = simpa_helper.extract_sinogram()

    intensity_ghost = np.random.random()
    offset = np.random.randint(10, 200)
    sino_ver_blur = nm.ghost_artifacts(sinogram, intensity_ghost, offset)

    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')

    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
    'path': path,
    'intensity ghost': intensity_ghost,
    'offset_ghost': offset
    }

    image_log.append(log)

    return data

def add_specle(path):
    print(f'### Specle is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()

    snr = np.random.randint(1, 10)
    
    sinogram = simpa_helper.extract_sinogram()
    sino_ver_blur = nm.speckle_noise(sinogram, snr)
    
    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')

    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
    'path': path,
    'snr': snr
    }

    image_log.append(log)

    return data

def add_blur_ghost(path):
    print(f'### Blur ghost is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    
    sinogram = simpa_helper.extract_sinogram()

    kernel_size = np.random.randint(3, 25)
    intensity_blur = np.random.random()
    sino_ver_blur = nm.random_blur(sinogram, kernel_size, intensity_blur)

    intensity_ghost = np.random.random()
    offset = np.random.randint(10, 200)

    sino_ver_blur = nm.ghost_artifacts(sino_ver_blur, intensity_ghost, offset)

    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')
    
    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
        "path": path,
        "kernel_size": kernel_size,
        "intensity_blur": intensity_blur,
        "intensity_ghost": intensity_ghost,
        "offset_ghost": offset
    }

    image_log.append(log)
    return data

def add_blur_noise(path):
    print(f'### Blur noise is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    
    sinogram = simpa_helper.extract_sinogram()

    kernel_size = np.random.randint(3, 25)
    intensity_blur = np.random.random()
    sino_ver_blur = nm.random_blur(sinogram, kernel_size, intensity_blur)

    snr = np.random.randint(1, 10)
    sino_ver_blur = nm.speckle_noise(sino_ver_blur, snr)
    
    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')
    
    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
        "path": path,
        "kernel_size": kernel_size,
        "intensity_blur": intensity_blur,
        "snr": snr
    }

    image_log.append(log)
    return data

def add_noise_ghost(path):
    print(f'### Noise ghost is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    
    sinogram = simpa_helper.extract_sinogram()

    intensity_ghost = np.random.random()
    offset = np.random.randint(10, 200)
    sino_ver_blur = nm.ghost_artifacts(sinogram, intensity_ghost, offset)

    snr = np.random.randint(1, 10)
    sino_ver_blur = nm.speckle_noise(sino_ver_blur, snr)

    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')
    
    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log = {
        "path": path,
        "intensity_ghost": intensity_ghost,
        "offset_ghost": offset,
        "snr": snr
    }

    image_log.append(log)

    return data

def add_ghost_blur_noise(path):
    print(f'### Blur ghost noise is added on {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    
    sinogram = simpa_helper.extract_sinogram()

    intensity_ghost = np.random.random()
    offset = np.random.randint(10, 200)
    sino_ver_blur = nm.ghost_artifacts(sinogram, intensity_ghost, offset)

    snr = np.random.randint(1, 10)
    sino_ver_blur = nm.speckle_noise(sino_ver_blur, snr)

    kernel_size = np.random.randint(3, 25)
    intensity_blur = np.random.random()
    sino_ver_blur = nm.random_blur(sino_ver_blur, kernel_size, intensity_blur)

    UID = uuid.uuid1()
    tmp_dir_path = os.path.join(tmpdir, f'tmp/tmp_{UID}.hdf5')
    
    # Save sinogram from the original file to a new hdf5 file to work from
    simpa_helper.save_sinogram(sino_ver_blur, tmp_dir_path)
    
    # Create new simpa helper instance with the new hdf5 file
    simpa_helper_noise = sh.SimpaHelper(tmp_dir_path)
    simpa_helper_noise.reconstruct()
    data = simpa_helper_noise.get_reconstructed_data()

    log =  {
        "path": path,
        "intensity_ghost": intensity_ghost,
        "offset_ghost": offset,
        "snr": snr,
        "kernel_size": kernel_size,
        "intensity_blur": intensity_blur
    }

    image_log.append(log)

    return data

def return_gt(path):
    print(f'### GT retrieved {path}')
    simpa_helper = sh.SimpaHelper(hdf5_file_path=path)
    simpa_helper.reconstruct()
    data = simpa_helper.get_reconstructed_data()
    return data

# Create train test val split on all files
filenames = os.listdir(os.path.join(tmpdir, 'simulated_raw_374/'))
train_filenames, test_filenames = train_test_split(filenames, test_size=0.15, random_state=42)
train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.1176, random_state=42)



src_dir = os.path.join(tmpdir, 'simulated_raw_374/')

dst_dirs = {
    'train': os.path.join(tmpdir, 'data_prepped/train/'),
    'val': os.path.join(tmpdir, 'data_prepped/val/'),
    'test': os.path.join(tmpdir, 'data_prepped/test/')
}

noise_functions = {
    'gt': return_gt,
    'blur': add_blur,
    'ghost_artifacts': add_specle,
    'noise': add_ghost,
    'blur_ghost_artifacts': add_blur_ghost,
    'blur_noise': add_blur_noise,
    'ghost_artifacts_noise': add_noise_ghost,
    'blur_ghost_artifacts_noise': add_ghost_blur_noise
}

classes = [
    'gt',
    'blur',
    'ghost_artifacts',
    'noise',
    'blur_ghost_artifacts',
    'blur_noise',
    'ghost_artifacts_noise',
    'blur_ghost_artifacts_noise'
]

os.makedirs(os.path.join(tmpdir, 'tmp/'), exist_ok=True)

try:
    wandb.init(
        project='DATASET CREATION',
        name=f'374_adapted_settings',
        config={
            'device': 'LedArray',
            'dataset': 'All',
        }
    )
except Exception as e:
    print(f'Error in wandDB {e}')

def apply_noise_and_save(src_dir, dst_dir, filename, noise_function):
    # Load the file
    filepath = os.path.join(src_dir, filename)

    noisy_image = noise_function(filepath)

    filename_without_ext, _ = os.path.splitext(filename)
    filename_npz = filename_without_ext + '.npz'

    # Save the noisy image in the destination directory
    dst_filepath = os.path.join(dst_dir, filename_npz)
    np.savez(dst_filepath, data=noisy_image)

class_to_int = {class_name: i for i, class_name in enumerate(classes)}
processed_files = 0

def worker(args):
    src_dir, dst_dir, filename, noise_function = args
    apply_noise_and_save(src_dir, dst_dir, filename, noise_function)

run = wandb.init(project="dataset_creation")

with Pool(32) as p:
    for class_id in classes:
        class_name = class_id
        noise_function = noise_functions[class_name]

        dst_dirs_class = {
            'train': os.path.join(dst_dirs['train'], str(class_to_int[class_name])),
            'val': os.path.join(dst_dirs['val'], str(class_to_int[class_name])),
            'test': os.path.join(dst_dirs['test'], str(class_to_int[class_name]))
        }

        for dst_dir in dst_dirs_class.values():
            os.makedirs(dst_dir, exist_ok=True)

        for dataset, filenames in [('train', train_filenames), ('val', val_filenames), ('test', test_filenames)]:
            p.map(worker, [(src_dir, dst_dirs_class[dataset], filename, noise_function) for filename in filenames])
            processed_files += len(filenames)
            wandb.log({"processed files": processed_files})


with open('log.json', 'w') as f:
    json.dump(list(image_log), f)

artifact = wandb.Artifact('image_log', type='dataset')
artifact.add_file('log.json')
wandb.log_artifact(artifact)
wandb.finish()