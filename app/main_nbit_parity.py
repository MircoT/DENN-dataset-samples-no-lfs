from data.creator import nbit_parity
from os import getcwd
import sys

def main(nbit= 3, training_size = None, test_size= None):
    fileout = 'n{}_bit_parity_db'.format(nbit)
    print("+++ START +++")
    print("+++")
    print("+++ output", fileout)
    print("+++")
    nbit_parity(fileout,
    nbit=nbit, 
    training_size=training_size, 
    test_size=test_size, 
    batch_size=2000, 
    dest_folder=getcwd()+'/../../DENN-dataset-nbitparty',
    depth=1)


if __name__ == '__main__':
    #args:
    try:
        nbit = int(sys.argv[1])
    except:
        nbit = 3
    try:
        training_size = int(sys.argv[2])
    except:
        training_size = 10000
    try:
        test_size = int(sys.argv[3])
    except:
        test_size = 1000
    #execute:
    main(nbit,training_size,test_size)
