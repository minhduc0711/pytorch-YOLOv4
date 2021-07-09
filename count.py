import os
import argparse as ap
import cv2
import numpy as np
from tqdm import tqdm

from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect

from sort import Sort


def filter_objects(boxes):
    # [bicycle, car, motorbike, bus, truck]
    included_classes = [1, 2, 3, 5, 7]
    keep = np.isin(boxes[:, 5].astype(np.int), included_classes)
    return boxes[keep]


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--cfg", type=str, default="./cfg/yolov4-tiny.cfg")
    parser.add_argument("--weights", type=str, default="./weights/yolov4-tiny.weights")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    model = Darknet(args.cfg)
    model.print_network()
    model.load_weights(args.weights)
    print(f'Loading weights from {args.weights}')
    if args.use_cuda:
        model.cuda()

    num_classes = model.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    tracker = Sort()
    vid = cv2.VideoCapture(args.video)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=num_frames)

    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video size", vw,vh)
        input_fname = os.path.basename(args.video)
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", input_fname.replace(".mp4", "_pred.mp4"))
        out_video = cv2.VideoWriter(output_path, fourcc, 20.0, (vw,vh))

    while True:
        ret, frame = vid.read()
        sized = cv2.resize(frame, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda=args.use_cuda)[0]
        boxes = filter_objects(np.array(boxes))
        tracked_boxes = tracker.update(boxes)

        res_img = plot_boxes_cv2(frame, tracked_boxes, class_names=class_names)
        count_str = ", ".join(
            [f"{class_names[k]}: {v}" for k, v in tracker.object_counts.items()]
        )
        res_img = cv2.putText(res_img, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        res_img = cv2.putText(res_img, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if args.save_output:
            out_video.write(res_img)
        else:
            cv2.imshow('frame', res_img)
            if cv2.waitKey(1) == ord('q'):
                break
        pbar.update(1)

    pbar.close()
    vid.release()
    cv2.destroyAllWindows()
    if args.save_output:
        out_video.release()
