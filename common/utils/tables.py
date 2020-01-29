TABLES = (
    {
        'label': '0001',
        'caption': '0001: Evaluation results on puppet dataset using default networks configurations',
        'tables': (
            ('Yolo 3', 'evaluation/eval-0001_Vanilla_Nets--Yolo3.csv'),
            ('F-RCNN', 'evaluation/eval-0001_Vanilla_Nets--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/eval-0001_Vanilla_Nets--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/eval-0001_Vanilla_Nets--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': 'the label',
        'caption': 'The Caption',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/eval-0033a_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
            ('RetinaNet 152 b', 'evaluation/eval-0033b_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
            ('RetinaNet 152 c', 'evaluation/eval-0033c_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
            ('RetinaNet 152 d', 'evaluation/eval-0033d_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
        ),
    },
)
