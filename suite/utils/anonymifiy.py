from argparse import ArgumentParser
from os.path import join
from shutil import copyfile
import hashlib
import os
import re


in_files = ()


def anonymize(source_dir: str, target_dir: str, case_id: str = None, print_translation: bool = False):
    """
    Anonymizes all files in a given ``source_dir`` with an md5 hash.
    Note that this script was implemented based on the assumption that every source file
    name contains a 10-digit case id that can be used to generate the hash.
    Adapt this for your own use-case.
    Alternatively, provide a case_id using the --case_id command line argument.

    :param print_translation: If True, prints the translation of the original filename to the anonymized filename
    :param case_id: Static case id to use (instead of 10-digit id from file name)
    :param source_dir: Dir in which to look for files recursively
    :param target_dir: Dir to which to save anonymized files
    """

    print('Anonymizing data from', source_dir)
    print('Collecting anonymized data in', target_dir)

    case_counter = 0
    case_counters = {}

    out_files = [''] * len(in_files)
    for path, dirs, files in os.walk(source_dir):

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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--case_id', required=False, default=None)
    parser.add_argument('--print_translation', required=False, action='store_true')

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    args.target_dir = os.path.abspath(args.target_dir)

    anonymize(args.base_dir, args.target_dir, args.case_id)
