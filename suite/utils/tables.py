TABLES = (

    # ===================
    # 1001 – Default Nets
    # ===================
    {
        'label': '1001',
        'caption': 'Training on Puppet Dataset (no transfer learning, no image augmentation), test on Puppet Dataset (env: 1001)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1001_Default_Nets--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1001_Default_Nets--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1001-body-shots',
        'caption': 'Training on Puppet Dataset (no transfer learning, no image augmentation), test on Public Images Dataset (env: 1001)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1001_Default_Nets--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1001_Default_Nets--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1001_Default_Nets--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ========================
    # 1002 – transfer learning
    # ========================
    {
        'label': '1002',
        'caption': 'Training on Puppet Dataset using transfer learning with frozen backbone (no image augmentation), test on Puppet Dataset (env: 1002)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1002_Transfer_Learning--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1002-body-shots',
        'caption': 'Training on Puppet Dataset using transfer learning with frozen backbone (no image augmentation), test on Public Images Dataset (env: 1002)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1002_Transfer_Learning--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1002_Transfer_Learning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =====================================
    # 1003 – transfer learning, Fine Tuning
    # =====================================
    {
        'label': '1003',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from transfer learning (no image augmentation), test on Puppet Dataset (env: 1003)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3.csv'),
            ('F-RCNN', 'evaluation/final/eval-1003_TL_Fine_Tuning--FRCNN.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1003-body-shots',
        'caption': 'Training on Puppet Dataset, fine-tuning backbone from transfer learning (no image augmentation), test on Public Images Dataset (env: 1003)',
        'tables': (
            ('Yolo 3', 'evaluation/final/eval-1002_Transfer_Learning--Yolo3-body-shots-eval.csv'),
            ('F-RCNN', 'evaluation/final/eval-1003_TL_Fine_Tuning--FRCNN-body-shots-eval.csv'),
            ('RetinaNet 50', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1003_TL_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ==========================
    # 1010a – image augmentation
    # ==========================
    {
        'label': '1010a',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no transfer learning), test on Puppet Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010a-body-shots',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no transfer learning), test on Public Images Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010a-all-cases',
        'caption': 'Training on Puppet Dataset, strong image augmentation (no transfer learning), test on Cases Dataset (env: 1010a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================
    # 1010b – Medium image augmentation
    # =================================
    {
        'label': '1010b',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no transfer learning), test on Puppet Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010b-body-shots',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no transfer learning), test on Public Images Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010b-all-cases',
        'caption': 'Training on Puppet Dataset, medium image augmentation (no transfer learning), test on Cases Dataset (env: 1010b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010b_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================
    # 1010c – weak image augmentation
    # ==============================
    {
        'label': '1010c',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no transfer Learning), test on Puppet Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1010c-body-shots',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no transfer Learning), test on Public Images Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1010c-all-cases',
        'caption': 'Training on Puppet Dataset, weak image augmentation (no transfer Learning), test on Cases Dataset (env: 1010c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1010c_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ============================================
    # 1011 – image augmentation, transfer learning
    # ============================================
    {
        'label': '1011',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (frozen backbone), test on Puppet Dataset (env: 1011)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011-body-shots',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (frozen backbone), test on Public Images Dataset (env: 1011)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    {
        'label': '1011-all-cases',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (frozen backbone), test on Cases Dataset (env: 1011)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ====================================================
    # 1011b – Medium image augmentation, transfer learning
    # ====================================================
    {
        'label': '1011b',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (frozen backbone), test on Puppet Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011b-body-shots',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (frozen backbone), test on Public Images Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1011b-all-cases',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (frozen backbone), test on Cases Dataset (env: 1011b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011b_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================================
    # 1011c – weak image augmentation, transfer learning
    # =================================================
    {
        'label': '1011c',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (frozen backbone), test on Puppet Dataset (env: 1011c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1011c-body-shots',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (frozen backbone), test on Public Images Dataset (env: 1011c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1011c-all-cases',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (frozen backbone), test on Cases Dataset (env: 1003)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1011c_Transfer_Learning_Image_Augmentation--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =========================================================
    # 1012 – image augmentation, transfer learning, Fine Tuning
    # =========================================================
    {
        'label': '1012',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (fine-tuning backbone), test on Puppet Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012-body-shots',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (fine-tuning backbone), test on Public Images Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012-all-cases',
        'caption': 'Training on Puppet Dataset, strong image augmentation with transfer learning (fine-tuning backbone), test on Cases Dataset (env: 1012a)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # =================================================================
    # 1012b – Medium image augmentation, transfer learning, Fine Tuning
    # =================================================================
    {
        'label': '1012b',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (fine-tuning backbone), test on Puppet Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012b-body-shots',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (fine-tuning backbone), test on Public Images Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012b-all-cases',
        'caption': 'Training on Puppet Dataset, medium image augmentation with transfer learning (fine-tuning backbone), test on Cases Dataset (env: 1012b)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012b_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },

    # ==============================================================
    # 1012c – weak image augmentation, transfer learning, Fine Tuning
    # ==============================================================
    {
        'label': '1012c',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), test on Puppet Dataset (env: 1012c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1012c-body-shots',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), test on Public Images Dataset (env: 1012c)',
        'tables': (
            ('RetinaNet 50', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 152', 'evaluation/final/eval-1012c_Transfer_Learning_Image_Augmentation_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1012c-all-cases',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), test on Cases Dataset (env: 1003)',
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
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Close-Ups Dataset (env: 1020, RetinaNet 50)',
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
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1020-retina50-body-shots-tiling',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
        )
    },
    {
        'label': '1020-retina50-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Cases Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1020-retina50-puppet-eval',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval.csv'),
        )
    },
    {
        'label': '1020-retina50-puppet-eval-tiling-800',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset, Tiling 800px (env: 1020, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
        )
    },
    {
        'label': '1020-retina152',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Close-Ups Dataset (env: 1020, RetinaNet 152)',
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
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1020-retina152-body-shots-tiling',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
        )
    },
    {
        'label': '1020-retina152-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Cases Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },
    {
        'label': '1020-retina152-puppet-eval',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval.csv'),
        )
    },
    {
        'label': '1020-retina152-puppet-eval-tiling-800',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset, Tiling 800px (env: 1020, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1020a_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1020b_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1020c_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1020d_Close_Up_Wounds--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
        )
    },

    # ===========================================
    # 1021 – Close Up Wounds Fine Tuning Training
    # ===========================================
    {
        'label': '1021-retina50',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Close-Ups Dataset (env: 1021, RetinaNet 50)',
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
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1021-retina50-body-shots-tiling',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiling800.csv'),
        )
    },
    {
        'label': '1021-retina50-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Cases Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-all-cases-eval.csv'),
        )
    },
    {
        'label': '1021-retina50-puppet-eval',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval.csv'),
        )
    },
    {
        'label': '1021-retina50-puppet-eval-tiling-800',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset with 800px tiling (env: 1021, RetinaNet 50)',
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet50-puppet-eval-tiling-800.csv'),
        )
    },
    {
        'label': '1021-retina152',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Close-Ups Dataset (env: 1021, RetinaNet 152)',
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
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1021-retina152-body-shots-tiling',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiling800.csv'),
        )
    },
    {
        'label': '1021-retina152-all-cases',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Cases Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-all-cases-eval.csv'),
        )
    },
    {
        'label': '1021-retina152-puppet-eval',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval.csv'),
        )
    },
    {
        'label': '1021-retina152-puppet-eval-tiling-800',
        'calculate_group_average': True,
        'caption': 'Training on Close-Ups Dataset, '
                   'strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset with 800px tiling (env: 1021, RetinaNet 152)',
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152-puppet-eval-tiling-800.csv'),
        )
    },


    # ==============================================================
    # 1030 – weak image augmentation, transfer learning, Fine Tuning
    # ==============================================================
    {
        'label': '1030-retina50',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
                   'test on Public Images Dataset (env: 1030, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1030-retina50-body-shots-eval-tiling800',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1030, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
        )
    },
    {
        'label': '1030-retina50-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
                   'test on Puppet Dataset and Close-Ups Dataset (env: 1030, Retina 152)',
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
                   'test on Public Images Dataset (env: 1030, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1030-retina152-body-shots-eval-tiling',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1030, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1030b_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
        )
    },
    {
        'label': '1030-retina152-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (frozen tuning backbone), '
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
    # 1031 – weak image augmentation, transfer learning, Fine Tuning
    # ==============================================================
    {
        'label': '1031-retina50',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset and Close-Ups Dataset (env: 1031, Retina 50)',
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1031, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1031-retina50-body-shots-eval-tiling',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1031, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval-tiled800.csv'),
        )
    },
    {
        'label': '1031-retina50-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset and Close-Ups Dataset (env: 1031, Retina 152)',
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
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1031, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    {
        'label': '1031-retina152-body-shots-eval-tiling',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset using 800px tiles (env: 1031, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval-tiled800.csv'),
        )
    },
    {
        'label': '1031-retina152-all-cases-eval',
        'caption': 'Training on Puppet Dataset and Close-Ups Dataset, image augmentation with transfer learning (fine-tuning backbone), '
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
    # 1032 – weak image augmentation, transfer learning, Fine Tuning
    # ==============================================================
    {
        'label': '1032-retina50',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
    #     'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
    #     'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
    # 1033 – weak image augmentation, transfer learning, Fine Tuning
    # ==============================================================
    {
        'label': '1033-retina50',
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1033, Retina 50)',
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
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
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
        'caption': 'Training on Puppet Dataset, weak image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1033, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1033a_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1033b_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1033c_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1033d_Joint_Puppet_Closeup_Wounds_Strong_Imgaug_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ===============================================================================
    # 1034 – weak image augmentation, Extracted Wounds, transfer learning
    # ===============================================================================
    {
        'label': '1034-retina50',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset (env: 1034, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1034a_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1034b_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1034c_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1034d_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1034-retina50-body-shots-eval',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning frozen backbone), '
                   'test on Public Images Dataset (env: 1034, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1034a_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1034b_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1034c_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1034d_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1034-retina152',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (frozen backbone), '
                   'test on Puppet Dataset (env: 1034, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1034a_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1034b_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1034c_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1034d_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1034-retina152-body-shots-eval',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1034, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1034a_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1034b_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1034c_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1034d_Joint_Puppet_Extracted_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ===============================================================================
    # 1035 – weak image augmentation, Extracted Wounds, transfer learning, Fine Tuning
    # ===============================================================================
    {
        'label': '1035-retina50',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset (env: 1035, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1035a_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1035b_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1035c_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1035d_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50.csv'),
        )
    },
    {
        'label': '1035-retina50-body-shots-eval',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1035, Retina 50)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 50 a', 'evaluation/final/eval-1035a_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 b', 'evaluation/final/eval-1035b_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 c', 'evaluation/final/eval-1035c_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
            ('RetinaNet 50 d', 'evaluation/final/eval-1035d_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50-body-shots-eval.csv'),
        )
    },
    {
        'label': '1035-retina152',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Puppet Dataset (env: 1035, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1035a_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1035b_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1035c_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1035d_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152.csv'),
        )
    },
    {
        'label': '1035-retina152-body-shots-eval',
        'caption': 'Training on Puppet Dataset with extracted wounds, Strong image augmentation with transfer learning (fine-tuning backbone), '
                   'test on Public Images Dataset (env: 1035, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1035a_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1035b_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1035c_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1035d_Joint_Puppet_Extracted_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================
    # 1100 – Training on Cases Dataset, Max 1
    # =======================================
    {
        'label': '1100-retina50',
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1100, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1100, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1100, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1100, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1101, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1101, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1101, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1101, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1110, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1110, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1110, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1110, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1111, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1111, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1111, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1111, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1112, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1112, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1112, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1112, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1113, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1113, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1113, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1113, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ==============================================================
    # 1200 – Training on Cases Dataset, Close-Ups Dataset (restricted)
    # ==============================================================
    {
        'label': '1200-retina50',
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1200, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1200, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1200, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1200, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # ===========================================================================
    # 1201 – Training on Cases Dataset, Close-Ups Dataset (restricted), Fine Tuning
    # ===========================================================================
    {
        'label': '1201-retina50',
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted), strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) (env: 1201, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1201, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1201, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1201, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
    
    # ==========================================================================================
    # 1300 – Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset
    # ==========================================================================================
    {
        'label': '1300-retina50',
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1300, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1300, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1300, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (frozen backbone), '
                   'test on Public Images Dataset (env: 1300, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },

    # =======================================================================================================
    # 1301 – Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, Fine Tuning
    # =======================================================================================================
    {
        'label': '1301-retina50',
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (fine-tuned backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1301, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (fine-tuned backbone), '
                   'test on Public Images Dataset (env: 1301, Retina 50)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (fine-tuned backbone), '
                   'test on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset (env: 1301, Retina 152)',
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
        'caption': 'Training on Cases Dataset, Close-Ups Dataset (restricted) and Close-Ups Dataset, strong image augmentation '
                   'with transfer learning (fine-tuned backbone), '
                   'test on Public Images Dataset (env: 1301, Retina 152)',
        'calculate_group_average': True,
        'tables': (
            ('RetinaNet 152 a', 'evaluation/final/eval-1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 b', 'evaluation/final/eval-1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 c', 'evaluation/final/eval-1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
            ('RetinaNet 152 d', 'evaluation/final/eval-1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet152-body-shots-eval.csv'),
        )
    },
)
