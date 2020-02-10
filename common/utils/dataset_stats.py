import json
from os.path import join

import numpy as np
import pandas as pd
from imgaug.parameters import show_distributions_grid, Clip, Normal, Absolute, Add
from matplotlib import pyplot as plt


DATA_PATH = 'data/vanilla_datasets/'
DATA_PATH = '/home/blsa/projects/confidential/wound-detection/data/'

DATASETS = (
    ('Puppet', 'puppet_measure_bands'),
    ('Full Body Shots', 'body_shots'),
    ('Close Up Wounds', 'closeup_wounds'),
    ('Close Up Wounds Cases', 'closeup_wounds_confidential'),
    ('Cases', 'cases-multishots'),
)

CLASS_CLUSTERS = {
    0: (
        1, 2, 3, 4, 5, 14
    ),
    1: (
        6, 7, 8, 9, 10, 11, 12, 13, 15
    )
}

if __name__ == '__main__':

    # show_distributions_grid([
    #     Add(Absolute(Normal(0.0, 3.0)), 1)
    # ],graph_sizes=(2048, 2048))

    index = pd.MultiIndex.from_tuples(
        (
            ('Images', '', 'n'),
            ('Images', '', 'Min/Max Width'),
            ('Images', '', 'Min/Max Height'),
            ('Images', '', 'Average Width'),
            ('Images', '', 'Average Height'),
            ('absolute', 'Sharp Force', 'n'),
            ('absolute', 'Sharp Force', 'Min/Max Width'),
            ('absolute', 'Sharp Force', 'Mean/Median Width'),
            ('absolute', 'Sharp Force', 'Min/Max Height'),
            ('absolute', 'Sharp Force', 'Mean/Median Height'),
            ('absolute', 'Sharp Force', 'Min/Max Area'),
            ('absolute', 'Sharp Force', 'Mean/Median Area'),

            ('absolute', 'Blunt Force', 'n'),
            ('absolute', 'Blunt Force', 'Min/Max Width'),
            ('absolute', 'Blunt Force', 'Mean/Median Width'),
            ('absolute', 'Blunt Force', 'Min/Max Height'),
            ('absolute', 'Blunt Force', 'Mean/Median Height'),
            ('absolute', 'Blunt Force', 'Min/Max Area'),
            ('absolute', 'Blunt Force', 'Mean/Median Area'),

            ('absolute', 'All Classes', 'n'),
            ('absolute', 'All Classes', 'Min/Max Width'),
            ('absolute', 'All Classes', 'Mean/Median Width'),
            ('absolute', 'All Classes', 'Min/Max Height'),
            ('absolute', 'All Classes', 'Mean/Median Height'),
            ('absolute', 'All Classes', 'Min/Max Area'),
            ('absolute', 'All Classes', 'Mean/Median Area'),

            ('relative', 'Sharp Force', 'n'),
            ('relative', 'Sharp Force', 'Min/Max Width'),
            ('relative', 'Sharp Force', 'Mean/Median Width'),
            ('relative', 'Sharp Force', 'Min/Max Height'),
            ('relative', 'Sharp Force', 'Mean/Median Height'),
            ('relative', 'Sharp Force', 'Min/Max Area'),
            ('relative', 'Sharp Force', 'Mean/Median Area'),

            ('relative', 'Blunt Force', 'n'),
            ('relative', 'Blunt Force', 'Min/Max Width'),
            ('relative', 'Blunt Force', 'Mean/Median Width'),
            ('relative', 'Blunt Force', 'Min/Max Height'),
            ('relative', 'Blunt Force', 'Mean/Median Height'),
            ('relative', 'Blunt Force', 'Min/Max Area'),
            ('relative', 'Blunt Force', 'Mean/Median Area'),

            ('relative', 'All Classes', 'n'),
            ('relative', 'All Classes', 'Min/Max Width'),
            ('relative', 'All Classes', 'Mean/Median Width'),
            ('relative', 'All Classes', 'Min/Max Height'),
            ('relative', 'All Classes', 'Mean/Median Height'),
            ('relative', 'All Classes', 'Min/Max Area'),
            ('relative', 'All Classes', 'Mean/Median Area'),
        )
    )

    dataframe = pd.DataFrame({
        dataset[0]: ('abc',) for dataset in DATASETS
    },
        index=index
    )

    for name, dataset_label in DATASETS:

        with open(join(DATA_PATH, dataset_label, 'annotations.json'), 'r', encoding='utf-8') as f:

            dataset = json.loads(''.join(f.readlines()))
            annotations = dataset.get('annotations', [])
            images = {}

            image_widths = []
            image_heights = []

            for image_data in dataset.get('images', []):
                images[image_data.get('id')] = image_data
                image_widths.append(image_data.get('width'))
                image_heights.append(image_data.get('height'))

            n_classes = np.empty(shape=(3, 1))
            widths = np.empty(shape=(3, 1), dtype='uint8')
            rel_widths = np.empty(shape=(3, 1), dtype='float32')
            heights = np.empty(shape=(3, 1), dtype='uint8')
            rel_heights = np.empty(shape=(3, 1), dtype='float32')
            areas = np.empty(shape=(3, 1), dtype='uint8')
            rel_areas = np.empty(shape=(3, 1), dtype='float32')

            for annotation in annotations:
                x, y, w, h = annotation.get('bbox', [])

                image_width = images.get(annotation.get('image_id')).get('width')
                image_height = images.get(annotation.get('image_id')).get('height')
                image_area = image_width * image_height

                class_idx = 0 if annotation.get('category_id') in CLASS_CLUSTERS[0] else 1

                n_classes[class_idx, 0] += 1
                n_classes[2, 0] += 1

                width_col = [[-1], [-1], [int(w)]]
                width_col[class_idx] = [int(w)]
                widths = np.append(widths, width_col, axis=1)

                heights_col = [[-1], [-1], [int(h)]]
                heights_col[class_idx] = [int(h)]
                heights = np.append(heights, heights_col, axis=1)

                areas_col = [[-1], [-1], [int(h) * int(w)]]
                areas_col[class_idx] = [int(h) * int(w)]
                areas = np.append(areas, areas_col, axis=1)

                width_col = [[-1], [-1], [int(w) / image_width]]
                width_col[class_idx] = [int(w) / image_width]
                rel_widths = np.append(rel_widths, width_col, axis=1)

                heights_col = [[-1], [-1], [int(h) / image_height]]
                heights_col[class_idx] = [int(h) / image_height]
                rel_heights = np.append(rel_heights, heights_col, axis=1)

                areas_col = [[-1], [-1], [int(h) * int(w) / image_area]]
                areas_col[class_idx] = [int(h) * int(w) / image_area]
                rel_areas = np.append(rel_areas, areas_col, axis=1)

            column = [''] * 47
            i = 0
            offsets = {
                'absolute': 0,
                'relative': 21,
                0: 0,
                1: 7,
                2: 14
            }

            widths = widths[:, 1:]
            widths = np.ma.masked_array(widths, widths == -1)

            rel_widths = rel_widths[:, 1:]
            rel_widths = np.multiply(np.ma.masked_array(rel_widths, rel_widths == -1.0), 100)

            heights = heights[:, 1:]
            heights = np.ma.masked_array(heights, heights == -1)

            rel_heights = rel_heights[:, 1:]
            rel_heights = np.ma.masked_array(rel_heights, rel_heights == -1) * 100

            areas = areas[:, 1:]
            areas = np.ma.masked_array(areas, areas == -1)

            rel_areas = rel_areas[:, 1:]
            rel_areas = np.ma.masked_array(rel_areas, rel_areas == -1) * 100

            column[0] = len(image_widths)
            column[1] = '{}/{}'.format(min(image_widths), max(image_widths))
            column[2] = '{}/{}'.format(min(image_heights), max(image_heights))
            column[3] = sum(image_widths) / len(image_widths)
            column[4] = sum(image_heights) / len(image_heights)

            for val_type in ('absolute', 'relative'):
                for class_name in (0, 1, 2):
                    offset = 5 + int(offsets[val_type] + offsets[class_name])
                    if val_type == 'absolute':
                        column[offset + 0] = n_classes[class_name][0]
                        column[offset + 1] = '{}/{}'.format(np.amin(widths[class_name]), np.amax(widths[class_name]))
                        column[offset + 2] = '{:.1f}/{}'.format(np.mean(widths[class_name]), np.ma.median(widths[class_name]))
                        column[offset + 3] = '{}/{}'.format(np.amin(heights[class_name]), np.amax(heights[class_name]))
                        column[offset + 4] = '{:.1f}/{}'.format(np.mean(heights[class_name]), np.ma.median(heights[class_name]))
                        column[offset + 5] = '{}/{}'.format(np.amin(areas[class_name]), np.amax(areas[class_name]))
                        column[offset + 6] = '{:.1f}/{}'.format(np.mean(areas[class_name]), np.ma.median(areas[class_name]))
                    else:
                        column[offset + 0] = '{:.2f}%'.format(n_classes[class_name][0] / n_classes[2][0] * 100)
                        column[offset + 1] = '{:.2f}%/{:.2f}%'.format(np.ma.min(rel_widths[class_name]), np.ma.amax(rel_widths[class_name]))
                        column[offset + 2] = '{:.2f}%/{:.2f}%'.format(np.mean(rel_widths[class_name]), np.ma.median(rel_widths[class_name]))
                        column[offset + 3] = '{:.2f}%/{:.2f}%'.format(np.amin(rel_heights[class_name]), np.amax(rel_heights[class_name]))
                        column[offset + 4] = '{:.2f}%/{:.2f}%'.format(np.mean(rel_heights[class_name]), np.ma.median(rel_heights[class_name]))
                        column[offset + 5] = '{:.2f}%/{:.2f}%'.format(np.amin(rel_areas[class_name]), np.amax(rel_areas[class_name]))
                        column[offset + 6] = '{:.2f}%/{:.2f}%'.format(np.mean(rel_areas[class_name]), np.ma.median(rel_areas[class_name]))

            dataframe.loc[:, name] = column

            fig = plt.figure(figsize=(7, 7))
            # plt.boxplot([widths[2], heights[2]], labels=('Widths', 'Heights'), widths=(0.5,0.5), )
            # plt.hist([widths[2], heights[2]], bins=30)
            plt.scatter(widths[2], heights[2])
            plt.legend(('widths', 'heights'))
            plt.xlabel('pixels')
            plt.ylabel('n boxes')
            # plt.grid()
            plt.show()

            with open('fig.pdf', 'wb') as f:
                fig.savefig(f, format='pdf')

    print(dataframe)
    print(dataframe.to_latex())
