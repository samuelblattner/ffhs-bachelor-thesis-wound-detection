from argparse import ArgumentParser
import os, re, hashlib
from os.path import join
from shutil import copyfile

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target_dir', required=True)

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    args.target_dir = os.path.abspath(args.target_dir)

    print('Anonymifying data from', args.base_dir)
    print('Collecting anonymified data in', args.target_dir)


    case_counter = 0
    case_counters = {}

    for path, dirs, files in os.walk(args.base_dir):

        case_id = re.search(r'(\d{10})', path)

        if case_id is None:
            continue

        case_hash = hashlib.md5(case_id.group(1).encode('utf-8')).hexdigest()
        case_counters.setdefault(case_hash, 0)

        for file in files:
            copyfile(join(path, file), join(args.target_dir, '{}-{}.jpg'.format(case_hash, case_counters[case_hash])))
            case_counters[case_hash] += 1