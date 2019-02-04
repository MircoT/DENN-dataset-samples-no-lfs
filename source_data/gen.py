import struct
from os import path
from random import randint

DATA_CIRCLE = [
    [   
        [0, 1, 1, 0], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ],
    [   
        [1, 1, 1, 1], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ],
    [   
        [1, 1, 1, 0], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [1, 1, 1, 0]
    ],
    [   
        [0, 1, 1, 1], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [0, 1, 1, 1]
    ],
    [   
        [1, 1, 1, 1], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ],
    [   
        [0, 1, 1, 0], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ],
    [   
        [0, 0, 0, 0], 
        [0, 1, 1, 0], 
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ],
    [   
        [0, 1, 1, 1], 
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]
]

DATA_LINE = [
    [   
        [1, 1, 1, 1], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ],
    [   
        [0, 0, 0, 0], 
        [1, 1, 1, 1], 
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ],
    [   
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [1, 1, 1, 1],
        [0, 0, 0, 0]
    ],
    [   
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ],
    [   
        [1, 0, 0, 0], 
        [1, 0, 0, 0], 
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ],
    [   
        [0, 1, 0, 0], 
        [0, 1, 0, 0], 
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ],
    [   
        [0, 0, 1, 0], 
        [0, 0, 1, 0], 
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ],
    [   
        [0, 0, 0, 1], 
        [0, 0, 0, 1], 
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ],
    [   
        [0, 0, 0, 1], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ],
    [   
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
]

CASUAL = []

for _ in range(6):
    cur_mat =[]
    for row in range(4):
        cur_mat.append([randint(0, 1) for column in range(4)])

    CASUAL.append(cur_mat)

DATA_CIRCLE_CLASS = [0 for _ in DATA_CIRCLE]
DATA_LINE_CLASS = [1 for _ in DATA_LINE]
CASUAL_CLASS = [randint(0, 1) for _ in CASUAL]

def write(name, data, label):
    """Write binary dataset

    Images:
        (uint) num images
        (uint) width images
        (uint) height images
        (uchar) images data ... 

    Labels:
        (uint) num labels
        (uchar) labels data ... 
    """
    with open(path.join('data', name+".img"), 'wb') as data_file:
        data_file.write(struct.pack('III', len(data), len(data[0]), len(data[0][0])))
        for elm in data:
            flat_elm = [int(item*255) for sublist in elm for item in sublist]
            data_file.write(struct.pack('B'*len(flat_elm), *flat_elm))

    with open(path.join('data', name+".lb"), 'wb') as data_file:
        data_file.write(struct.pack('I', len(label)))
        data_file.write(struct.pack('B'*len(label), *label))

write('images', 
    DATA_CIRCLE+DATA_LINE+CASUAL, 
    DATA_CIRCLE_CLASS+DATA_LINE_CLASS+CASUAL_CLASS
    )