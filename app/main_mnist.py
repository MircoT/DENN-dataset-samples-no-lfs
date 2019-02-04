from data.creator import mnist
from os import getcwd


def main():
    mnist('test', batch_size=2000, dest_folder=getcwd()+'/../../DENN-LITE-rnn/', depth=2)


if __name__ == '__main__':
    main()
