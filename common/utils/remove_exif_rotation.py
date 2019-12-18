import os
from argparse import ArgumentParser

from PIL import Image, ExifTags, ImageOps
from os.path import join

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--target_dir', required=True)

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    args.target_dir = os.path.abspath(args.target_dir)

    for path, dirs, files in os.walk(args.base_dir):

        for file in files:

            try:
                im = Image.open(join(path, file))
            except OSError:
                continue

            exif = {
                ExifTags.TAGS[k]: v
                for k, v in im._getexif().items()
                if k in ExifTags.TAGS
            }

            orientation = exif.get('Orientation')

            angle = 0
            mirror = 0

            if orientation in (5, 6):
                angle = -90

            elif orientation in (7, 8):
                angle = 90

            elif orientation in (3, 4):
                angle = -180

            if orientation in (2, 4, 5, 7):
                mirror = 1

            im = im.rotate(angle, expand=1)
            if mirror > 0:
                im = ImageOps.mirror(im)

            im.save(join(args.target_dir, file), 'JPEG')