import argparse
import csv
import requests
import os.path
import tqdm
from multiprocessing import Pool
from functools import partial


def load_index(index_file_name, classes=None):
    print('loading image index from {}'.format(index_file_name))
    index = []
    with open(index_file_name, 'r') as index_file:
        reader = csv.reader(index_file)
        header = next(reader)
        for row in reader:
            id = row[0]
            url = row[1]
            image_class = row[2]
            if not classes or image_class  in classes:
                index.append((id, url))
    return index

def download(output_path, args):
    id, url = args
    if url == 'None':
        return
    r = requests.get(url)
    if r.status_code == 200:
        with open(os.path.join(output_path, id), 'wb') as f:
            f.write(r.content)

def download_files(index, output_path, workers):
    print('downloading image to {}'.format(output_path))
    func = partial(download, output_path)
    worker_pool = Pool(workers)
    for _ in tqdm.tqdm(worker_pool.imap(func, index), total=len(index)):
        pass
    worker_pool.close()
    worker_pool.join()


def Main():
    parser = argparse.ArgumentParser(description='Download images in dataset')
    parser.add_argument('-i', '--index_file', required=True)
    parser.add_argument('-o', '--output_path', default='./')
    parser.add_argument('-w', '--workers', default=1, type=int)
    parser.add_argument('-c', '--classes')
    args = parser.parse_args()
    index = load_index(args.index_file, args.classes)
    download_files(index, args.output_path, args.workers)


if __name__ == "__main__":
    Main()
