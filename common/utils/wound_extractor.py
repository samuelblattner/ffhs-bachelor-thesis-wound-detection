import os
from argparse import ArgumentParser
from os.path import join
import json
import numpy as np

from PIL import Image

MARGINS = [50, 100, 200]


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    os.makedirs(join(args.output_dir, 'images'), exist_ok=True)

    with open(join(args.dataset_dir, 'annotations.json'), 'r', encoding='utf-8') as a:

        dataset = json.loads(''.join(a.readlines()))

        # Output dataset
        output_dataset = {}
        image_id_counter = 0
        annot_id_count = 0

        # 1. Collect annotations and group by image
        annotations_per_image = {}
        for annotation in dataset.get('annotations', []):
            annotations_per_image.setdefault(annotation.get('image_id'), []).append(annotation)

        # 2. Iterate over every image and crop on every annotation
        for image_data in dataset.get('images', []):

            base_image = np.asarray(Image.open(join(args.dataset_dir, image_data.get('path'))), dtype='uint8')

            for ann, annotation in enumerate(annotations_per_image.get(image_data.get('id'), [])):
                for margin in MARGINS:

                    image = base_image.copy()

                    x, y, w, h = annotation.get('bbox')
                    x1 = int(x - margin)
                    y1 = int(y - margin)
                    x2 = int(x + w + margin)
                    y2 = int(y + h + margin)

                    cropped_image_width = x2 - x1
                    cropped_image_height = y2 - y1

                    pad_left = abs(min(x1, 0))
                    pad_right = max(0, x2 - image.shape[1])
                    pad_top = abs(min(y1, 0))
                    pad_bottom = max(0, y2 - image.shape[0])

                    image = np.pad(image, (
                        (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)
                    ), mode='constant')

                    cropped_wound = image[y1:y2, x1:x2, :]
                    filename = '{}-w{}-{}.jpg'.format(
                        image_data.get('file_name').replace('.jpg', ''),
                        '{}'.format(ann).zfill(5),
                        margin
                    )

                    Image.fromarray(cropped_wound).save(join(args.output_dir, 'images', filename), quality=100)

                    output_dataset.setdefault('images', []).append({
                        'id': image_id_counter,
                        'path': 'images/{}'.format(filename),
                        'width': cropped_wound.shape[1],
                        'height': cropped_wound.shape[0],
                        'file_name': filename
                    })

                    # 3. Iterate over all annotations contained in the image and
                    # add those that overlap with the current image crop

                    for crop_ann, crop_annotation in enumerate(annotations_per_image.get(image_data.get('id'), [])):

                        abs_cx, abs_cy, _, __ = crop_annotation.get('bbox')

                        cx1 = min(cropped_image_width, max(0, int(abs_cx - x1)))
                        cy1 = min(cropped_image_height, max(0, int(abs_cy - y1)))
                        cx2 = min(cropped_image_width, max(0, int(abs_cx + w - x1)))
                        cy2 = min(cropped_image_height, max(0, int(abs_cy + h - y1)))

                        # Discard zero width or zero height boxes
                        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
                            continue

                        # 4. Crop segmentation if any
                        segs = crop_annotation.get('segmentation', [])
                        cropped_segs = []

                        for seg in segs:

                            cropped_seg = []
                            for sx, sy in zip(seg[0::2], seg[1::2]):
                                # TODO: Cut out-of-frame masks properly
                                cropped_seg.append(round(min(cropped_image_width, max(0, sx - x1)), 2))
                                cropped_seg.append(round(min(cropped_image_height, max(0, sy - y1)), 2))

                            cropped_segs.append(cropped_seg)

                        output_dataset.setdefault('annotations', []).append({
                            'id': annot_id_count,
                            'image_id': image_id_counter,
                            'category_id': annotation.get('category_id'),
                            'bbox': [margin, margin, w, h],
                            'segmentation': cropped_segs
                        })
                        annot_id_count += 1

                    image_id_counter += 1

        output_dataset.setdefault('categories', dataset.get('categories'))

        with open(join(args.output_dir, 'annotation.json'), 'w', encoding='utf-8') as o:
            o.write(json.dumps(output_dataset))
