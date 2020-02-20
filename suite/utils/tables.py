TABLES = (

    # ===================
    # 1001 – Default Nets
    # ===================
    {
        'label': '1001',
        'caption': 'Training on Puppet Dataset (default network configurations, no image augmentation), test on Puppet Dataset (env: 1001)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1001_Default_Nets--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1001_Default_Nets--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1001-body-shots',
        'caption': 'Training on Puppet Dataset (default network configurations, no image augmentation), test on Full Body Shots Dataset (env: 1001)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1001_Default_Nets--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1001_Default_Nets--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ========================
    # 1002 – Transfer Learning
    # ========================
    {
        'label': '1002',
        'caption': 'Training on Puppet Dataset using Transfer Learning with frozen Backbone (no image augmentation), test on Puppet Dataset (env: 1002)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1002_Transfer_Learning--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1002-body-shots',
        'caption': 'Training on Puppet Dataset using Transfer Learning with frozen Backbone (no image augmentation), test on Full Body Shots Dataset (env: 1002)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1002_Transfer_Learning--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =====================================
    # 1003 – Transfer Learning, Fine Tuning
    # =====================================
    {
        'label': '1003',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on Puppet Dataset (env: 1003)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1003_TL_Fine_Tuning--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1003-body-shots',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on Full Body Shots Dataset (env: 1003)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1003_TL_Fine_Tuning--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ==========================
    # 1010a – Image Augmentation
    # ==========================
    {
        'label': '1010a',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no Transfer Learning), test on Puppet Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010a-body-shots',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no Transfer Learning), test on Full Body Shots Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010a-all-cases',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no Transfer Learning), test on All Cases Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================
    # 1010b – Medium Image Augmentation
    # =================================
    {
        'label': '1010b',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no Transfer Learning), test on Puppet Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010b-body-shots',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no Transfer Learning), test on Full Body Shots Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010b-all-cases',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no Transfer Learning), test on All Cases Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================
    # 1010c – Low Image Augmentation
    # ==============================
    {
        'label': '1010c',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no Transfer Learning), test on Puppet Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010c-body-shots',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no Transfer Learning), test on Full Body Shots Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010c-all-cases',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no Transfer Learning), test on All Cases Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ============================================
    # 1011 – Image Augmentation, Transfer Learning
    # ============================================
    {
        'label': '1011',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (frozen Backbone), test on Puppet Dataset (env: 1011)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011-body-shots',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (frozen Backbone), test on Full Body Shots Dataset (env: 1011)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    {
        'label': '1010-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1003)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ====================================================
    # 1011b – Medium Image Augmentation, Transfer Learning
    # ====================================================
    {
        'label': '1011b',
        'caption': 'Training on Puppet Dataset, Medium Image Augmentation with Transfer Learning (frozen Backbone), test on Puppet Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011b-body-shots',
        'caption': 'Training on Puppet Dataset, Medium Image Augmentation with Transfer Learning (frozen Backbone), test on Full Body Shots Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1011b-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================================
    # 1011c – Low Image Augmentation, Transfer Learning
    # =================================================
    {
        'label': '1011c',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (frozen Backbone), test on Puppet Dataset (env: 1011c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011c-body-shots',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (frozen Backbone), test on Full Body Shots Dataset (env: 1011c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1011c-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1003)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =========================================================
    # 1012 – Image Augmentation, Transfer Learning, Fine Tuning
    # =========================================================
    {
        'label': '1012',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (fine tuning Backbone), test on Puppet Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012-body-shots',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (fine tuning Backbone), test on Full Body Shots Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================================================
    # 1012b – Medium Image Augmentation, Transfer Learning, Fine Tuning
    # =================================================================
    {
        'label': '1012b',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (fine tuning Backbone), test on Puppet Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012b-body-shots',
        'caption': 'Training on Puppet Dataset, Image Augmentation with Transfer Learning (fine tuning Backbone), test on Full Body Shots Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012b-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================================================
    # 1012c – Low Image Augmentation, Transfer Learning, Fine Tuning
    # ==============================================================
    {
        'label': '1012c',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), test on Puppet Dataset (env: 1012c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012c-body-shots',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), test on Full Body Shots Dataset (env: 1012c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012c-all-cases',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from Transfer Learning (no image augmentation), test on All Cases Dataset (env: 1003)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ===============================
    # 1020 – Close Up Wounds Training
    # ===============================
    {
        'label': '1020-retina50',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Close Up Wounds Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1020-retina50-body-shots',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Full Body Shots Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1020-retina50-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Cases Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1020-retina152',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Close Up Wounds Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1020-retina152-body-shots',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Full Body Shots Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1020-retina152-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Cases Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ===========================================
    # 1021 – Close Up Wounds Fine Tuning Training
    # ===========================================
    {
        'label': '1021-retina50',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Close Up Wounds Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1021-retina50-body-shots',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Full Body Shots Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1021-retina50-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Cases Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1021-retina152',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Close Up Wounds Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1021-retina152-body-shots',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Full Body Shots Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1021-retina152-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close Up Wounds Dataset, '
                   'strong image augmentation with Transfer Learning (frozen backbone), '
                   'test on Cases Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },



    # ==============================================================
    # 1030 – Low Image Augmentation, Transfer Learning, Fine Tuning
    # ==============================================================
    {
        'label': '1030-retina50',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset (env: 1030, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1030-retina50-body-shots-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1030, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1030-retina50-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Cases Dataset (env: 1030, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1030-retina152',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset and Close Up Wounds Dataset (env: 1030, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1030-retina152-body-shots-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1030, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1030-retina152-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Cases Dataset (env: 1030, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================================================
    # 1031 – Low Image Augmentation, Transfer Learning, Fine Tuning
    # ==============================================================
    {
        'label': '1031-retina50',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset and Close Up Wounds Dataset (env: 1031, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1031-retina50-body-shots-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1031, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1031-retina50-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Cases Dataset (env: 1031, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1031-retina152',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset and Close Up Wounds Dataset (env: 1031, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1031-retina152-body-shots-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1031, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1031-retina152-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close Up Wounds Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Cases Dataset (env: 1031, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================================================
    # 1032 – Low Image Augmentation, Transfer Learning, Fine Tuning
    # ==============================================================
    {
        'label': '1032-retina50',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset (env: 1032, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1032a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1032b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1032c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1032d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50.csv'),
        )
    },
    # {
    #     'label': '1032-retina50-body-shots-eval',
    #     'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
    #                'test on Puppet Dataset (env: 1032, Retina 50)',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 50 a', 'evaluation/final/eval-1032a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50-body-shots-eval.csv'),
    #         ('RetinaNet 50 b', 'evaluation/final/eval-1032b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50-body-shots-eval.csv'),
    #         ('RetinaNet 50 c', 'evaluation/final/eval-1032c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50-body-shots-eval.csv'),
    #         ('RetinaNet 50 d', 'evaluation/final/eval-1032d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet50-body-shots-eval.csv'),
    #     )
    # },
    {
        'label': '1032-retina152',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset (env: 1032, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1032a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1032b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1032c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1032d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152.csv'),
        )
    },
    # {
    #     'label': '1032-retina152-body-shots-eval',
    #     'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
    #                'test on Puppet Dataset (env: 1032, Retina 152)',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/final/eval-1032a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152-body-shots-eval.csv'),
    #         ('RetinaNet 152 b', 'evaluation/final/eval-1032b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152-body-shots-eval.csv'),
    #         ('RetinaNet 152 c', 'evaluation/final/eval-1032c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152-body-shots-eval.csv'),
    #         ('RetinaNet 152 d', 'evaluation/final/eval-1032d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug--RetinaNet-Resnet152-body-shots-eval.csv'),
    #     )
    # },

    # ==============================================================
    # 1033 – Low Image Augmentation, Transfer Learning, Fine Tuning
    # ==============================================================
    {
        'label': '1033-retina50',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset (env: 1033, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1033a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1033b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1033c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1033d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1033-retina50-body-shots-eval',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1033, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1033a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1033b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1033c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1033d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1033-retina152',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Puppet Dataset (env: 1033, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1033a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1033b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1033c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1033d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1033-retina152-body-shots-eval',
        'caption': 'Training on Puppet Dataset, Low Image Augmentation with Transfer Learning (fine tuning Backbone), '
                   'test on Full Body Shots Dataset (env: 1033, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1033a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1033b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1033c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1033d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================
    # 1100 – Training on Cases Dataset, Max 1
    # =======================================
    {
        'label': '1100-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1100, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1100a_Cases--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1100b_Cases--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1100c_Cases--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1100d_Cases--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1100-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1100, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1100a_Cases--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1100b_Cases--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1100c_Cases--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1100d_Cases--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1100-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1100, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1100a_Cases--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1100b_Cases--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1100c_Cases--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1100d_Cases--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1100-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1100, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1100a_Cases--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1100b_Cases--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1100c_Cases--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1100d_Cases--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ====================================================
    # 1101 – Training on Cases Dataset, Max 1, Fine Tuning
    # ====================================================
    {
        'label': '1101-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1101, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1101a_Cases_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1101b_Cases_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1101c_Cases_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1101d_Cases_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1101-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1101, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1101a_Cases_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1101b_Cases_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1101c_Cases_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1101d_Cases_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1101-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1101, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1101a_Cases_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1101b_Cases_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1101c_Cases_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1101d_Cases_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1101-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1101, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1101a_Cases_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1101b_Cases_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1101c_Cases_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1101d_Cases_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================
    # 1110 – Training on Cases Dataset, Max 3
    # =======================================
    {
        'label': '1110-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1110, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1110a_Cases_Multishot_3--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1110b_Cases_Multishot_3--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1110c_Cases_Multishot_3--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1110d_Cases_Multishot_3--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1110-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1110, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1110a_Cases_Multishot_3--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1110b_Cases_Multishot_3--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1110c_Cases_Multishot_3--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1110d_Cases_Multishot_3--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1110-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1110, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1110a_Cases_Multishot_3--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1110b_Cases_Multishot_3--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1110c_Cases_Multishot_3--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1110d_Cases_Multishot_3--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1110-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1110, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1110a_Cases_Multishot_3--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1110b_Cases_Multishot_3--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1110c_Cases_Multishot_3--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1110d_Cases_Multishot_3--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================
    # 1111 – Training on Cases Dataset, Max 3
    # =======================================
    {
        'label': '1111-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1111, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1111-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1111, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1111-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1111, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1111-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1111, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    
    
    # =======================================
    # 1112 – Training on Cases Dataset, Max 6
    # =======================================
    {
        'label': '1112-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1112, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1112a_Cases_Multishot_6--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1112b_Cases_Multishot_6--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1112c_Cases_Multishot_6--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1112d_Cases_Multishot_6--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1112-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1112, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1112a_Cases_Multishot_6--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1112b_Cases_Multishot_6--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1112c_Cases_Multishot_6--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1112d_Cases_Multishot_6--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1112-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1112, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1112a_Cases_Multishot_6--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1112b_Cases_Multishot_6--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1112c_Cases_Multishot_6--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1112d_Cases_Multishot_6--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1112-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1112, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1112a_Cases_Multishot_6--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1112b_Cases_Multishot_6--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1112c_Cases_Multishot_6--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1112d_Cases_Multishot_6--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ====================================================
    # 1113 – Training on Cases Dataset, Max 6, Fine Tuning
    # ====================================================
    {
        'label': '1113-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1113, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1113-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1113, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1113-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1113, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1113-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1113, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ==============================================================
    # 1200 – Training on Cases Dataset, Close Up Wounds Confidential
    # ==============================================================
    {
        'label': '1200-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1200, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1200-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1200, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1200-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1200, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1200-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1200, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ===========================================================================
    # 1201 – Training on Cases Dataset, Close Up Wounds Confidential, Fine Tuning
    # ===========================================================================
    {
        'label': '1201-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential (env: 1201, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1201-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1201, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1201-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1201, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1201-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1201, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    
    # ==========================================================================================
    # 1300 – Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset
    # ==========================================================================================
    {
        'label': '1300-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1300, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1300-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1300, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1300-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1300, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1300-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (frozen Backbone), '
                   'test on Full Body Shots Dataset (env: 1300, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================================================================================
    # 1301 – Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, Fine Tuning
    # =======================================================================================================
    {
        'label': '1301-retina50',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (fine-tuned Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1301, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1301-retina50-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (fine-tuned Backbone), '
                   'test on Full Body Shots Dataset (env: 1301, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1301-retina152',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (fine-tuned Backbone), '
                   'test on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset (env: 1301, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1301-retina152-body-shots-eval',
        'caption': 'Training on Cases Dataset, Close Up Wounds Confidential and Close Up Wounds Dataset, strong image augmentation '
                   'with Transfer Learning (fine-tuned Backbone), '
                   'test on Full Body Shots Dataset (env: 1301, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },


    #
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, puppet',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, Puppet Dataset',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, puppet, fullsize',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, Puppet Dataset, Fullsize',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, body shots',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, Body Shot Dataset',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3-Body-Shots-1.csv'),
    #         ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN--Body-Shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50-body-shots.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, body shots, fullsize',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, Body Shot Dataset, Fullsize',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3-Body-Shots-1.csv'),
    #         ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN--Body-Shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50-body-shots-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, all cases',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, All Cases',
    #     'tables': (
    #         # ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3-Body-Shots-1.csv'),
    #         # ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN--Body-Shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50-all-cases.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0004-image-augmentation-scale-1x-6x, all cases',
    #     'caption': '0004: Configuration 0004: Image Augmentation Scale 1x to 6x, All Cases, Fullsize',
    #     'tables': (
    #         # ('Yolo 3', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--Yolo3-Body-Shots-1.csv'),
    #         # ('F-RCNN', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--FRCNN--Body-Shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet50-all-cases-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0004_Image_Augmentation_Scale_1x_to_6x--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0005-transfer-learning',
    #     'caption': '0005: Transfer Learning',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0005_Transfer_Learning--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0005_Transfer_Learning--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0005-transfer-learning-fullsize',
    #     'caption': '0005: Transfer Learning',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0005_Transfer_Learning--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0005_Transfer_Learning--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet50-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0005-transfer-learning-body-shots',
    #     'caption': '0005: Transfer Learning',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0005_Transfer_Learning--Yolo3-body-shots.csv'),
    #         ('F-RCNN', 'evaluation/eval-0005_Transfer_Learning--FRCNN-body-shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet50-body-shots.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # # {
    # #     'label': '0005-transfer-learning-body-shots',
    # #     'caption': '0005: Transfer Learning',
    # #     'tables': (
    # #         ('Yolo 3', 'evaluation/eval-0005_Transfer_Learning--Yolo3-body-shots.csv'),
    # #         ('F-RCNN', 'evaluation/eval-0005_Transfer_Learning--FRCNN-body-shots.csv'),
    # #         ('RetinaNet 50', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet50-body-shots-fullsize.csv'),
    # #         ('RetinaNet 152', 'evaluation/eval-0005_Transfer_Learning--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    # #     )
    # # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x',
    #     'caption': '0006: Transfer Learning',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x-fullsize',
    #     'caption': '0006: Transfer Learning Fullsize',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x-body-shots',
    #     'caption': '0006: Transfer Learning Body Shots',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3-body-shots.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN-body-shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50--body-shots.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x-body-shots-fullsize',
    #     'caption': '0006: Transfer Learning Body Shots Fullsize',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3-body-shots.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN-body-shots.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50--body-shots-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152--body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x-body-shots-allcases',
    #     'caption': '0006: Transfer Learning Body Shots All Cases',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3-all-cases.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN-all-cases.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-all-cases.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0006-tf-imgaug-scale-1x-7x-body-shots-allcases-fullsize',
    #     'caption': '0006: Transfer Learning Body Shots All Cases Fullsize',
    #     'tables': (
    #         ('Yolo 3', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--Yolo3-all-cases.csv'),
    #         ('F-RCNN', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--FRCNN-all-cases.csv'),
    #         ('RetinaNet 50', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-all-cases-fullsize.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0006_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-all-cases-full-size.csv'),
    #     )
    # },
    # {
    #     'label': '0007-tf-imgaug-scale-1x-7x-rot89',
    #     'caption': '0007: Transfer Learning Scale rot',
    #     'tables': (
    #         ('RetinaNet 50', 'evaluation/eval-0007_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0007_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0007-tf-imgaug-scale-1x-7x-rot89-fullsize',
    #     'caption': '0007: Transfer Learning Scale rot fullsize',
    #     'tables': (
    #         ('RetinaNet 50', 'evaluation/eval-0007_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50-fullsize.csv'),
    #         (
    #             'RetinaNet 152',
    #             'evaluation/eval-0007_Transfer_Learning_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0007c-ultrahires-tf-imgaug-scale-1x-7x-rot89',
    #     'caption': '0007c: Transfer Learning Scale rot ultrahires',
    #     'tables': (
    #         ('RetinaNet 50',
    #          'evaluation/eval-0007c_Transfer_Learning_UltraHires_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50.csv'),
    #     )
    # },
    # {
    #     'label': '0007c-ultrahires-tf-imgaug-scale-1x-7x-rot89-fullsize',
    #     'caption': '0007c: Transfer Learning Scale rot ultrahires',
    #     'tables': (
    #         ('RetinaNet 50',
    #          'evaluation/eval-0007c_Transfer_Learning_UltraHires_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50-fullsi6ze.csv'),
    #     )
    # },
    # {
    #     'label': '0007c-ultrahires-tf-imgaug-scale-1x-7x-rot89-all-cases',
    #     'caption': '0007c: Transfer Learning Scale rot all cases',
    #     'tables': (
    #         ('RetinaNet 50',
    #          'evaluation/eval-0007c_Transfer_Learning_UltraHires_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0007c-ultrahires-tf-imgaug-scale-1x-7x-rot89-all-cases-fullsize',
    #     'caption': '0007c: Transfer Learning Scale rot all cases fullsize',
    #     'tables': (
    #         ('RetinaNet 50',
    #          'evaluation/eval-0007c_Transfer_Learning_UltraHires_Image_Augmentation_Scale_1x_to_7x_rotate90_1000_Epochs--RetinaNet-Resnet50-all-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0007e-tl-hires',
    #     'caption': '0007e: TF HIRES',
    #     'tables': (
    #         ('RetinaNet 50', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50.csv'),
    #         ('RetinaNet 50, Fullsize', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-fullsize.csv'),
    #         ('RetinaNet 152, Fullsize', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0007e-tl-hires-body-shots',
    #     'caption': '0007e: TF HIRES Body shots',
    #     'tables': (
    #         ('RetinaNet 50', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-body-shots.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 50, Fullsize',
    #          'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152, Fullsize',
    #          'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0007e-tl-hires-all-cases',
    #     'caption': '0007e: TF HIRES Allcases',
    #     'tables': (
    #         ('RetinaNet 50', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet50-all-cases.csv'),
    #         ('RetinaNet 152', 'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 50, Fullsize',
    #          'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152, Fullsize',
    #          'evaluation/eval-0007e_Transfer_Learning_Hires_Image_Augmentation_Scale_1x_to_7x--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training',
    #     'caption': '0011: All Cases Training',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-fullsize',
    #     'caption': '0011: All Cases Training Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-body-shots',
    #     'caption': '0011: All Cases Training Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-body-shots-fullsize',
    #     'caption': '0011: All Cases Training Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-1024',
    #     'caption': '0011: All Cases Training 1024',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_1024--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_1024--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_1024--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_1024--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-1024-fullsize',
    #     'caption': '0011: All Cases Training 1024 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_1024--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_1024--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_1024--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_1024--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-1024-body-shots',
    #     'caption': '0011: All Cases Training 1024 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_1024--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-1024-body-shots-fullsize',
    #     'caption': '0011: All Cases Training 1024 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3',
    #     'caption': '0011: All Cases Training Max-3',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3-fullsize',
    #     'caption': '0011: All Cases Training Max-3 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3-body-shots',
    #     'caption': '0011: All Cases Training Max-3 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3-body-shots-fullsize',
    #     'caption': '0011: All Cases Training Max-3 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3-all-cases',
    #     'caption': '0011: All Cases Training Max-3 All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-3-all-cases-fullsize',
    #     'caption': '0011: All Cases Training Max-3 All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6',
    #     'caption': '0011: All Cases Training Max-6',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6-fullsize',
    #     'caption': '0011: All Cases Training Max-6 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6-body-shots',
    #     'caption': '0011: All Cases Training Max-6 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6-body-shots-fullsize',
    #     'caption': '0011: All Cases Training Max-6 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6-all-cases',
    #     'caption': '0011: All Cases Training Max-6 All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0011-all-cases-training-Max-6-all-cases-fullsize',
    #     'caption': '0011: All Cases Training Max-6 All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0011_All_Cases_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0011b_All_Cases_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0011c_All_Cases_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0011d_All_Cases_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0014
    # {
    #     'label': '0014-all-cases-training-puppetbase',
    #     'caption': '0014: All Cases Training Puppetbase',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-body-shots',
    #     'caption': '0014: All Cases Training Puppetbase Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-body-shots-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-1024',
    #     'caption': '0014: All Cases Training Puppetbase 1024',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-1024-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase 1024 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-1024-body-shots',
    #     'caption': '0014: All Cases Training Puppetbase 1024 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-1024-body-shots-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase 1024 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_1024--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3',
    #     'caption': '0014: All Cases Training Puppetbase Max-3',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-3 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3-body-shots',
    #     'caption': '0014: All Cases Training Puppetbase Max-3 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3-body-shots-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-3 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3-all-cases',
    #     'caption': '0014: All Cases Training Puppetbase Max-3 All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-3-all-cases-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-3 All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_3--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6',
    #     'caption': '0014: All Cases Training Puppetbase Max-6',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-6 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6-body-shots',
    #     'caption': '0014: All Cases Training Puppetbase Max-6 Body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6-body-shots-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-6 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6-all-cases',
    #     'caption': '0014: All Cases Training Puppetbase Max-6 All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0014-all-cases-training-puppetbase-Max-6-all-cases-fullsize',
    #     'caption': '0014: All Cases Training Puppetbase Max-6 All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0014_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0014b_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0014c_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0014d_All_Cases_Transfer_ImgAug_Puppetbase_Max_6--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0017
    # # ====
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90-fullsize',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90 body shots',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90 body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90-body-shots-fullsize',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90 all cases',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90 all cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0017-closeup-wounds-confidential-1x-3x-rot90-all-cases-fullsize',
    #     'caption': '0017: Close up wounds confidential 1x 3x rot90 all cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017b_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017c_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017d_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0017e
    # # =====
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90-fullsize',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90 body shots',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90 body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90-body-shots-fullsize',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90 all cases',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90 all cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0017e-closeup-wounds-confidential-1x-3x-rot90-all-cases-fullsize',
    #     'caption': '0017e: Close up wounds confidential 1x 3x rot90 all cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017e_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017f_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         # ('RetinaNet 152 c', 'evaluation/eval-0017g_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017h_tl_imgaug_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0017i
    # # =====
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #         # ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90-fullsize',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90 Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #         # ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90 body shots',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90 body shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90-body-shots-fullsize',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90 Body shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90 all cases',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90 all cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0017icloseup-wounds-confidential-1x-3x-rot90-all-cases-fullsize',
    #     'caption': '0017i: Close up wounds confidential 1x 3x rot90 all cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0017i_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0017j_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0017k_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0017l_tl_imgaug_1x_3x_rot90_closeup_wounds_confidential--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0031
    # # ====
    # {
    #     'label': '0031 Joint Training Puppet Cases',
    #     'caption': '0031: Joint Training Puppet Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152.csv'),
    #     )
    # },
    # {
    #     'label': '0031 Joint Training Puppet Cases Fullsize',
    #     'caption': '0031: Joint Training Puppet Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0031 Joint Training Puppet Cases Body Shots',
    #     'caption': '0031: Joint Training Puppet Cases Body Shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0031 Joint Training Puppet Cases Body Shots Fullsize',
    #     'caption': '0031: Joint Training Puppet Cases Body Shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0031 Joint Training Puppet Cases All Cases',
    #     'caption': '0031: Joint Training Puppet Cases All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0031 Joint Training Puppet Cases All Cases Fullsize',
    #     'caption': '0031: Joint Training Puppet Cases All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0031a_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0031b_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0031c_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0031d_Joint_Training_of_Puppet_and_Cases--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0032
    # # ====
    # {
    #     'label': '0032 Joint Training Puppet Cases Body Shots',
    #     'caption': '0032: Joint Training Puppet Cases Body Shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0032a_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0032b_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0032c_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0032d_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0032 Joint Training Puppet Cases Body Shots Fullsize',
    #     'caption': '0032: Joint Training Puppet Cases Body Shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0032a_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0032b_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0032c_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0032d_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0032 Joint Training Puppet Cases All Cases',
    #     'caption': '0032: Joint Training Puppet Cases All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0032a_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0032b_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0032c_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0032d_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0032 Joint Training Puppet Cases All Cases Fullsize',
    #     'caption': '0032: Joint Training Puppet Cases All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0032a_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0032b_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0032c_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0032d_Joint_Training_of_Puppet_and_Cases_and_Closeups--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },
    #
    # # 0033
    # # ====
    # {
    #     'label': '0033 Joint Training Puppet Cases Body Shots',
    #     'caption': '0033: Joint Training Puppet Cases Body Shots',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0033a_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0033b_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0033c_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0033d_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots.csv'),
    #     )
    # },
    # {
    #     'label': '0033 Joint Training Puppet Cases Body Shots Fullsize',
    #     'caption': '0033: Joint Training Puppet Cases Body Shots Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a',
    #          'evaluation/eval-0033a_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 b',
    #          'evaluation/eval-0033b_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 c',
    #          'evaluation/eval-0033c_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #         ('RetinaNet 152 d',
    #          'evaluation/eval-0033d_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-body-shots-fullsize.csv'),
    #     )
    # },
    # {
    #     'label': '0033 Joint Training Puppet Cases All Cases',
    #     'caption': '0033: Joint Training Puppet Cases All Cases',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a', 'evaluation/eval-0033a_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 b', 'evaluation/eval-0033b_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 c', 'evaluation/eval-0033c_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases.csv'),
    #         ('RetinaNet 152 d', 'evaluation/eval-0033d_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases.csv'),
    #     )
    # },
    # {
    #     'label': '0033 Joint Training Puppet Cases All Cases Fullsize',
    #     'caption': '0033: Joint Training Puppet Cases All Cases Fullsize',
    #     'calculate_group_average': True,
    #     'tables': (
    #         ('RetinaNet 152 a',
    #          'evaluation/eval-0033a_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 b',
    #          'evaluation/eval-0033b_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 c',
    #          'evaluation/eval-0033c_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #         ('RetinaNet 152 d',
    #          'evaluation/eval-0033d_Joint_Training_of_Puppet_and_Cases_and_Closeups_and_Closeupsconf--RetinaNet-Resnet152-all-cases-fullsize.csv'),
    #     )
    # },

)
