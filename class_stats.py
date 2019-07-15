import argparse
import csv
from collections import Counter

def Main():
    parser = argparse.ArgumentParser(description='Count class stats in dataset')
    parser.add_argument('-i', '--index_file', required=True)
    args = parser.parse_args()

    print('loading image index from {}'.format(args.index_file))
    stats = Counter()
    with open(args.index_file, 'r') as index_file:
        reader = csv.reader(index_file)
        header = next(reader)
        for row in reader:
            if not row[2] or row[1] == 'None':
                continue
            stats[row[2]] += 1
    print(stats.most_common(50))


if __name__ == "__main__":
    Main()
