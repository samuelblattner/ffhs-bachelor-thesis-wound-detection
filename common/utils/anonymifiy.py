from argparse import ArgumentParser
import os, re, hashlib
from os.path import join
from shutil import copyfile

in_files = (

)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--case_id', required=False, default=None)
    parser.add_argument('--print_translation', required=False, action='store_true')

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    args.target_dir = os.path.abspath(args.target_dir)

    print('Anonymifying data from', args.base_dir)
    print('Collecting anonymified data in', args.target_dir)

    case_counter = 0
    case_counters = {}

    out_files = [''] * len(in_files)
    for path, dirs, files in os.walk(args.base_dir):

        case_id = re.search(r'(\d{10})', path)

        if case_id is None:
            if args.case_id is None:
                continue
            else:
                case_id = args.case_id
        else:
            case_id = case_id.group(1)

        case_hash = hashlib.md5(case_id.encode('utf-8')).hexdigest()
        case_counters.setdefault(case_hash, 0)

        for file in files:
            target_file = '{}-{}.jpg'.format(case_hash, case_counters[case_hash])

            if in_files:
                out_files[in_files.index(file[:file.index('.')])] = target_file

            if args.print_translation:
                print('{} ---> {}'.format(
                    file, target_file
                ))
            copyfile(join(path, file), join(args.target_dir, target_file))
            case_counters[case_hash] += 1

    print('\n'.join(out_files))