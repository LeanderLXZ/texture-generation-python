from multiprocessing import Pool
import os
from os.path import join
from texture_synthesis import TextureSynthesis

def worker(x):
    file_name, texture_path, save_path = x[0], x[1], x[2]
    TextureSynthesis(file_name=file_name,
                     texture_path=texture_path,
                     save_path=save_path,
                     input_size=[100, 100],
                     window_size=[15, 15],
                     output_size=[100, 100],
                     apply_gaussian=True).synthesis()

params = []
classes = os.listdir('data/dtd')
for class_name in classes:
    files = os.listdir(join('data/dtd', class_name))
    for file in files:
        if file[-4:] == '.jpg':
            params.append((file, join(join('data/dtd', class_name), file), join('results/', class_name)))

with Pool(len(params)) as p:
    p.map(worker, params)
