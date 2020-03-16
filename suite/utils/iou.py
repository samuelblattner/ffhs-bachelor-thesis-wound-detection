from neural_nets.retina_net.keras_retinanet.utils.compute_overlap import compute_overlap
import numpy as np

boxes = (
    (100, 100, 214, 163),
    (100.0, 100.0, 100.0, 100.0),
)

query_boxes = (
    (66, 66, 214, 163),
    (88, 88, 214 * 1.41, 163 * 1.41),
    (120, 80, 214 * 0.67, 163 * 1.5),
    (120, 120, 214 * 0.6, 163 * 0.84),
    (50., 50., 100.0, 100.0),
)

# print(list([float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])] for b in boxes))
# print(list([float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])] for b in query_boxes))
ov = compute_overlap(
    np.array([
        [float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])] for b in boxes
    ]),
    np.array([
        [float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])] for b in query_boxes
    ])
)

print(ov)