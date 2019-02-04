import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import datasetgenerator


class GenCommunicator(QObject):

    signal_done = pyqtSignal()
    step = pyqtSignal([int])


class Generator(QThread, QObject):

    def __init__(self, comm, gen_args, gen_kwargs):
        super(self.__class__, self).__init__()
        self.__args = gen_args
        self.__kwargs = gen_kwargs
        self.__comm = comm
        self.__exit = False

    def stop(self):
        self.__exit = True

    def __del__(self):
        self.wait()

    def run(self):
        # your logic here
        for step in datasetgenerator.new(*self.__args, UI=True, **self.__kwargs):
            self.__comm.step.emit(step)
            if self.__exit:
                return
        self.__comm.signal_done.emit()


class DatasetGenerator(QWidget):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.title = 'DENN Dataset Generator'
        self.width = 640
        self.height = 480
        self.__gen_thread = None

        self.__sources = [
            "./source_data/bezdekIris",
            "./source_data/letter-recognition",
            "./source_data/MNIST",
            "./source_data",
            "./source_data/magic",
            "./source_data/qsar",
            "./source_data/PimaIndiansDiabetes",
            "./source_data/BreastCancerWisconsin",
            "./source_data/CHD"
        ]
        self.__loaders = {
            "iris": datasetgenerator.IrisLoader(),
            "letter": datasetgenerator.LetterLoader(),
            "mnist": datasetgenerator.MNISTLoader(),
            "mnist in [0.1, 1.0]": datasetgenerator.MNISTLoader_01_10_Norm(),
            "bank": datasetgenerator.BankLoader(),
            "magic": datasetgenerator.MagicLoader(),
            "qsar": datasetgenerator.QsarLoader(),
            "coil": datasetgenerator.CoilLoader(),
            "coil_28x28": datasetgenerator.CoilWithResizeLoader(),
            "GeccoFirstDomain": datasetgenerator.GeccoFirstDomain(),
            "GasSensorArrayDrift": datasetgenerator.GasSensorArrayDrift(),
            "PimaIndiansDiabetes": datasetgenerator.PimaIndiansDiabetes(),
            "WisconsinDiagnosticBreastCancer": datasetgenerator.WisconsinDiagnosticBreastCancer(),
            "HeartDisease": datasetgenerator.HeartDisease()
        }
        self.__types = [
            "float",
            "double"
        ]
        self.__ds_versions = [
            "1",
            "2"
        ]
        self.__extraction_types = [
            "equal_division",
            "random",
            "respect_ratio"
        ]
        ##
        # Item percentages
        self.__train_elm_percentage = 1.0
        self.label_train_item_percentage = QLabel("{:.02f} %".format(100))
        self.__test_elm_percentage = 1.0
        self.label_test_item_percentage = QLabel("{:.02f} %".format(100))
        ##
        # Create UI
        self.initUI()
        ##
        # Events
        self.__gen_comm = GenCommunicator()
        self.__gen_comm.signal_done.connect(self.gen_done)
        self.__gen_comm.step.connect(self.update_pbar)

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.resize(self.width, self.height)

        ##
        # Source
        self.label_source = QLabel("Source")
        self.combo_source = QComboBox(self)

        for source in sorted(self.__sources):
            self.combo_source.addItem(source)

        ##
        # Set MNIST as default
        self.combo_source.setCurrentIndex(1)

        ##
        # Loader
        self.label_loader = QLabel("Loader")
        self.combo_loader = QComboBox(self)

        for loader in sorted(self.__loaders):
            self.combo_loader.addItem(loader)

        ##
        # Set MNIST as default
        self.combo_loader.setCurrentIndex(7)

        ##
        # Out name
        self.label_out_name = QLabel("Out base name")
        self.textbox_out_name = QLineEdit(self)

        ##
        # Dataset version
        self.label_ds_version = QLabel("Dataset version")
        self.combo_ds_version = QComboBox(self)

        for version in sorted(self.__ds_versions):
            self.combo_ds_version.addItem(version)

        ##
        # Size
        self.label_size = QLabel("Size")
        self.textbox_size = QLineEdit(self)

        ##
        # Batch size checkbox
        self.cb_batch_size = QCheckBox('Is batch size')
        self.cb_batch_size.toggle()

        ##
        # N shuffle
        self.label_n_shuffles = QLabel("N. shuffles")
        self.textbox_n_shuffles = QLineEdit(self)
        self.textbox_n_shuffles.setText("1")

        ##
        # Seed
        self.label_seed = QLabel("Seed")
        self.textbox_seed = QLineEdit(self)
        self.textbox_seed.setText("None")

        ##
        # Type
        self.label_type = QLabel("Type")
        self.combo_type = QComboBox(self)

        for type_ in sorted(self.__types):
            self.combo_type.addItem(type_)

        ##
        # Set float as default
        self.combo_type.setCurrentIndex(1)

        ##
        # Train %
        self.label_train_p = QLabel("Training")
        self.textbox_train_p = QLineEdit(self)
        self.textbox_train_p.setText("0.8")

        ##
        # Validation %
        self.label_validation_p = QLabel("Validation")
        self.textbox_validation_p = QLineEdit(self)
        self.textbox_validation_p.setText("0.1")

        ##
        # Test %
        self.label_test_p = QLabel("Test")
        self.textbox_test_p = QLineEdit(self)
        self.textbox_test_p.setText("0.1")

        ##
        # Test %
        self.label_cross_k = QLabel("K")
        self.textbox_cross_k = QLineEdit(self)
        self.textbox_cross_k.setText("0")
        self.textbox_cross_k.setValidator(QIntValidator())

        ##
        # Test %
        self.label_cross_i = QLabel("i")
        self.textbox_cross_i = QLineEdit(self)
        self.textbox_cross_i.setText("0")
        self.textbox_cross_i.setValidator(QIntValidator())

        ##
        # Autobalance
        self.cb_autobalance = QCheckBox('Autobalance')

        ##
        # Extraction type
        self.label_extraction_type = QLabel("Batch extraction type")
        self.combo_extraction_type = QComboBox(self)

        for type_ in sorted(self.__extraction_types):
            self.combo_extraction_type.addItem(type_)

        ##
        # Generate Button
        self.button_generate = QPushButton("Generate")
        self.button_generate.clicked.connect(self.on_generate)

        ##
        # Progress bar
        self.pbar = QProgressBar()
        self.pbar.setValue(0)

        ##
        # Item percentage
        self.sld_train_items = QSlider(Qt.Horizontal, self)
        self.sld_train_items.setFocusPolicy(Qt.NoFocus)
        self.sld_train_items.valueChanged[int].connect(
            self.on_change_train_slider)
        self.sld_train_items.setMinimum(1)
        self.sld_train_items.setMaximum(1000)
        self.sld_train_items.setValue(1000)
        self.sld_train_items.setTickInterval(1)
        self.label_sld_train = QLabel("Train Num. items")

        self.sld_test_items = QSlider(Qt.Horizontal, self)
        self.sld_test_items.setFocusPolicy(Qt.NoFocus)
        self.sld_test_items.valueChanged[int].connect(
            self.on_change_test_slider)
        self.sld_test_items.setMinimum(1)
        self.sld_test_items.setMaximum(1000)
        self.sld_test_items.setValue(1000)
        self.sld_test_items.setTickInterval(1)
        self.label_sld_test = QLabel("Test Num. items")

        ##
        # Layout
        hbox_0 = QHBoxLayout()
        hbox_0.addWidget(self.label_source)
        hbox_0.addWidget(self.combo_source)
        hbox_0.addWidget(self.label_loader)
        hbox_0.addWidget(self.combo_loader)

        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(self.label_out_name)
        hbox_1.addWidget(self.textbox_out_name)
        hbox_1.addWidget(self.label_ds_version)
        hbox_1.addWidget(self.combo_ds_version)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(self.label_size)
        hbox_2.addWidget(self.textbox_size)
        hbox_2.addWidget(self.cb_batch_size)

        hbox_3 = QHBoxLayout()
        hbox_3.addWidget(self.label_n_shuffles)
        hbox_3.addWidget(self.textbox_n_shuffles)
        hbox_3.addWidget(self.label_seed)
        hbox_3.addWidget(self.textbox_seed)
        hbox_3.addWidget(self.label_type)
        hbox_3.addWidget(self.combo_type)

        hbox_4 = QHBoxLayout()
        hbox_4.addWidget(self.label_train_p)
        hbox_4.addWidget(self.textbox_train_p)
        hbox_4.addWidget(self.label_validation_p)
        hbox_4.addWidget(self.textbox_validation_p)
        hbox_4.addWidget(self.label_test_p)
        hbox_4.addWidget(self.textbox_test_p)

        hbox_5 = QVBoxLayout()
        hbox_5_1  = QHBoxLayout()
        hbox_5_1.addWidget(QLabel("Cross-validation"))
        hbox_5_2  = QHBoxLayout()
        hbox_5_2.addWidget(self.label_cross_k)
        hbox_5_2.addWidget(self.textbox_cross_k)
        hbox_5_2.addWidget(self.label_cross_i)
        hbox_5_2.addWidget(self.textbox_cross_i)
        hbox_5.addLayout(hbox_5_1)
        hbox_5.addLayout(hbox_5_2)

        hbox_6 = QHBoxLayout()
        hbox_6.addWidget(self.cb_autobalance)
        hbox_6.addWidget(self.label_extraction_type)
        hbox_6.addWidget(self.combo_extraction_type)

        hbox_7 = QHBoxLayout()
        hbox_7.addWidget(self.label_sld_train)
        hbox_7.addWidget(self.sld_train_items)
        hbox_7.addWidget(self.label_train_item_percentage)

        hbox_8 = QHBoxLayout()
        hbox_8.addWidget(self.label_sld_test)
        hbox_8.addWidget(self.sld_test_items)
        hbox_8.addWidget(self.label_test_item_percentage)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_0)
        vbox.addLayout(hbox_1)
        vbox.addLayout(hbox_2)
        vbox.addLayout(hbox_3)
        vbox.addLayout(hbox_4)
        vbox.addLayout(hbox_5)
        vbox.addLayout(hbox_6)
        vbox.addLayout(hbox_7)
        vbox.addLayout(hbox_8)
        vbox.addWidget(self.button_generate)
        vbox.addWidget(self.pbar)

        self.setLayout(vbox)

        self.show()

    def on_change_train_slider(self, value):
        self.__train_elm_percentage = value / 1000.
        self.label_train_item_percentage.setText(
            "{:.02f} %".format(self.__train_elm_percentage * 100.))

    def on_change_test_slider(self, value):
        self.__test_elm_percentage = value / 1000.
        self.label_test_item_percentage.setText(
            "{:.02f} %".format(self.__test_elm_percentage * 100.))

    def on_generate(self):
        if self.__gen_thread is None:
            try:
                gen_args = [
                    self.combo_source.currentText(),
                    self.__loaders[self.combo_loader.currentText()],
                    int(self.textbox_size.text()),
                    self.textbox_out_name.text()
                ]
                gen_kwargs = {
                    'n_shuffle': int(self.textbox_n_shuffles.text()),
                    'batch_size': self.cb_batch_size.isChecked(),
                    'seed': int(self.textbox_seed.text()
                                ) if self.textbox_seed.text() != "None" else None,
                    'train_percentage': float(self.textbox_train_p.text()),
                    'validation_percentage': float(self.textbox_validation_p.text()),
                    'test_percentage': float(self.textbox_test_p.text()),
                    'autobalance': self.cb_autobalance.isChecked(),
                    'type_': self.combo_type.currentText(),
                    'extraction_type': self.combo_extraction_type.currentText(),
                    'train_perc_items': self.__train_elm_percentage,
                    'test_perc_items': self.__test_elm_percentage,
                    'version': int(self.combo_ds_version.currentText()),
                    'k_i_fold': {
                        'k':int(self.textbox_cross_k.text()),
                        'i':int(self.textbox_cross_i.text())
                        }
                }
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "!!! Attention !!!",
                    "One or more parameters are wrong...\n" + str(e),
                    QMessageBox.Ok
                )
            else:
                self.__gen_thread = Generator(
                    self.__gen_comm,
                    gen_args,
                    gen_kwargs
                )
                self.button_generate.setText("Stop")
                self.pbar.setValue(0)
                self.__gen_thread.start()
        else:
            self.__gen_thread.stop()
            self.__gen_thread = None
            self.button_generate.setText("Generate")

    def gen_done(self):
        self.__gen_thread = None
        self.button_generate.setText("Generate")
        QMessageBox.information(
            self,
            "Info",
            "Dataset complete!",
            QMessageBox.Close
        )

    def update_pbar(self, step):
        self.pbar.setValue(step)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DatasetGenerator()
    sys.exit(app.exec_())
