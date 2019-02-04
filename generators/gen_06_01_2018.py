import sys
sys.path.insert(0, "..")

from app.data.creator import *
from os import path, makedirs

DEST_FOLDER = path.join("DATASETS", "d_06_01_2018")


def main():
    dest_folder = path.join("..", DEST_FOLDER)
    makedirs(dest_folder, exist_ok=True)

    mnist("mnist_balanced_classes_normalized", dest_folder=dest_folder, normalized=True, batch_size=100, balanced_classes=True)
    mnist("mnist_balanced_classes", dest_folder=dest_folder, normalized=False, batch_size=100, balanced_classes=True)
    mnist("mnist_normalized", dest_folder=dest_folder, normalized=True, batch_size=100)
    mnist("mnist", dest_folder=dest_folder, normalized=False, batch_size=100, save_stats=True)

    fashion_mnist("fashion-mnist_balanced_classes_normalized", dest_folder=dest_folder, normalized=True, batch_size=100, balanced_classes=True)
    fashion_mnist("fashion-mnist_balanced_classes", dest_folder=dest_folder, normalized=False, batch_size=100, balanced_classes=True)
    fashion_mnist("fashion-mnist_normalized", dest_folder=dest_folder, normalized=True, batch_size=100)
    fashion_mnist("fashion-mnist", dest_folder=dest_folder, normalized=False, batch_size=100, save_stats=True)

    gas_sensor_array_drift("gass_balanced_classes_normalized", dest_folder=dest_folder, normalized=True, batch_size=10, balanced_classes=True)
    gas_sensor_array_drift("gass_balanced_classes", dest_folder=dest_folder, normalized=False, batch_size=10, balanced_classes=True)
    gas_sensor_array_drift("gass_normalized", dest_folder=dest_folder, normalized=True, batch_size=10)
    gas_sensor_array_drift("gass", dest_folder=dest_folder, normalized=False, batch_size=10, save_stats=True)

    magic_gamma_telescope("magic_balanced_classes_normalized", dest_folder=dest_folder, normalized=True, batch_size=10, balanced_classes=True)
    magic_gamma_telescope("magic_balanced_classes", dest_folder=dest_folder, normalized=False, batch_size=10, balanced_classes=True)
    magic_gamma_telescope("magic_normalized", dest_folder=dest_folder, normalized=True, batch_size=10)
    magic_gamma_telescope("magic", dest_folder=dest_folder, normalized=False, batch_size=10, save_stats=True)

    qsar("qsar_balanced_classes_normalized", dest_folder=dest_folder, normalized=True, batch_size=10, balanced_classes=True)
    qsar("qsar_balanced_classes", dest_folder=dest_folder, normalized=False, batch_size=10, balanced_classes=True)
    qsar("qsar_normalized", dest_folder=dest_folder, normalized=True, batch_size=10)
    qsar("qsar", dest_folder=dest_folder, normalized=False, batch_size=10, save_stats=True)


if __name__ == '__main__':
    main()
