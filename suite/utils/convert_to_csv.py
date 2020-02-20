import json
from os.path import join

if __name__ == '__main__':

    base = 'data/vanilla_datasets/puppet_measure_bands/'
    image_paths = {}
    category_names = {}

    with open(join(base, 'annotations.json'), 'r', encoding='utf-8') as f:
        dataset = json.loads(''.join(f.readlines()))
        annotations = dataset.get('annotations', [])
        images = dataset.get('images')
        categories = dataset.get('categories')

        for image in images:
            image_paths.setdefault(image.get('id'), image.get('path'))

        for category in categories:
            category_names.setdefault(category.get('id'), category.get('name'))

        with open(join(base, 'annotations.csv'), 'w', encoding='utf-8') as g:


            for annotation in annotations:
                x, y, w, h = annotation.get('bbox', [])
                g.write('{},{},{},{},{},{}\n'.format(
                    join(image_paths.get(annotation.get('image_id'))),
                    int(float(x)), int(float(y)), int(x + w), int(y + h), category_names.get(annotation.get('category_id'))
                ))
