import os
from argparse import ArgumentParser
from os.path import join

from PIL import Image

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--case_id', required=False, default=None)
    parser.add_argument('--print_translation', required=False, action='store_true')

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    args.target_dir = os.path.abspath(args.target_dir)

    for path, dirs, files in os.walk(args.base_dir):

        print(path)
        for file in files:
            if '.jpg' in file:
                print(file)
                # file = '0b15db873691dd5d2c208d67cd5decb7-5.jpg'
                im = Image.open(join(path, file))
                # h = int(800/3456 * 5184)
                h = int(0.5 * 5184)
                w = int(0.5 * 3456)
                im = im.resize((w, h))
                # im.show()
                im.save(join(args.target_dir, file), quality=100)
                # exit()
            # im.save()

    # print('\n'.join(out_files))