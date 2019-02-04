import datasetgenerator


def main():

    datasets = [
        # ("./source_data/bezdekIris", datasetgenerator.IrisLoader(), 6, "iris", 5, True, "double"),
        # ("./source_data/letter-recognition", datasetgenerator.LetterLoader(), 20, "letter", 5, True, "double"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 200, "mnist", 5, False, "double"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 1000, "mnist", 5, False, "double"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 2000, "mnist", 5, False, "double"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 4000, "new_mnist_d_5v", 5, True, "double"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 50, "mnist_d", 5, True, "double"),
        # FLOAT
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 2000, "mnist_f", 5, False, "float"),
        # ("./source_data/MNIST", datasetgenerator.MNISTLoader(), 4000, "mnist_f", 5, False, "float"),
        # ("./source_data", datasetgenerator.BankLoader(), 2000, "bank_f_no_duration", 5, False, "float"),
        # ("./source_data", datasetgenerator.BankLoader(), 2000, "bank_d_no_duration", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 2000, "magic_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 100, "qsar_d", 5, False, "double"),
        # ("./source_data", datasetgenerator.BankLoader(), 4000, "bank_d_no_duration_0", 5, False, "double"),
        # ("./source_data", datasetgenerator.BankLoader(), 4000, "bank_d_no_duration_1", 5, False, "double"),
        # ("./source_data", datasetgenerator.BankLoader(), 4000, "bank_d_no_duration_2", 5, False, "double"),
        # ("./source_data", datasetgenerator.BankLoader(), 4000, "bank_d_no_duration_3", 5, False, "double"),
        # ("./source_data", datasetgenerator.BankLoader(), 4000, "bank_d_no_duration_4", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 100, "test_magic_100_d", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 200, "test_magic_200_d", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 400, "test_magic_400_d", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 800, "test_magic_800_d", 5, False, "double"),
        # ("./source_data/magic", datasetgenerator.MagicLoader(), 1600, "test_magic_1600_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 10, "test_qsar_10_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 20, "test_qsar_20_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 40, "test_qsar_40_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 80, "test_qsar_80_d", 5, False, "double"),
        # ("./source_data/qsar", datasetgenerator.QsarLoader(), 160, "test_qsar_160_d", 5, False, "double"),
        # ("./source_data", datasetgenerator.CoilLoader(), 10, "coil-proc-28x28", 1, False, "float"),
        ("./source_data", datasetgenerator.GeccoFirstDomain(), 200, "first_domain_grid", 1, True, "float"),

    ]

    for dataset, loader, size, out_name, n_shuffle, batch_size, type_ in datasets:
        for _ in datasetgenerator.new(dataset, loader, size, out_name, n_shuffle=n_shuffle, batch_size=batch_size, type_=type_,
                                      train_percentage=0.80, validation_percentage=0.10, autobalance=False):
            pass

if __name__ == '__main__':
    main()
