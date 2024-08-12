import argparse
import logging
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import sys
import torch
import time
import av
from threading import Thread
import threading as th
from queue import Queue, Empty, Full

sys.path.append("yolov5")
from detection.detection_utils.detector import Detector
from utilities.homography import HomographyTransformer
from utilities.logging_config import DEFAULT_FORMAT

ENCLOSURE_SHAPE = np.array([4609, 3275])


def read_video(video_file: str,
               batch_size: int,
               q: Queue,
               c_q: Queue,
               tb_log: str,
               select: int = 1):
    """Thread target for a single video

    Args:
        video_file (str): Path to video file
        batch_size (int): Batch size to pack for
        q (Queue): Queue to put images etc
        c_q (Queue): Queue to listen to for stop orders
        tb_log (str): Path to tensorboard log
        select (int): Forward every <select>th frame 
    """
    logger = logging.getLogger(th.current_thread().name)
    logger.debug("Starting video thread")

    try:
        cap = av.open(video_file)
    except Exception as e:
        logger.warning("Could not open video {}. Aborting. Error: {}".format(video_file, e.msg))
        exit()
    enc_part = int(os.path.basename(video_file).split("_")[0].replace("ch", "")) - 1

    # precompute for tb logging
    max_frames = (cap.duration * 12) / 1000000
    max_steps = float(max_frames)
    step_size = np.ceil(max_steps / 100)
    video_num = th.current_thread().name.split(" ")[-1]

    # calculate start time of video
    start_time = os.path.basename(video_file).split("_")[1].split(".mp4")[0]
    start_time = time.strptime(start_time, "%Y%m%d%H%M%S")
    start_time = time.mktime(start_time)

    # correct internal frame time with stream start time
    stream = cap.streams.video[0]
    start_time -= float(stream.start_time * stream.time_base)

    logger.debug("Opened video {}".format(video_file))

    # set up tensorboard log
    if tb_log is not None:
        tb_writer = SummaryWriter(os.path.join(tb_log, video_num))

    # counter = 0
    next_step = 0
    step_counter = 0
    st = time.time()

    img_batch = []
    time_stamps = []

    cap.streams.video[0].thread_type = 'AUTO'

    for counter, f in enumerate(cap.decode(video=0)):
        # log at every full % made
        if counter >= next_step and tb_log is not None:
            time_taken = time.time() - st
            tb_writer.add_scalar("Progress", int(video_num), step_counter)
            tb_writer.add_scalar("Mean FPS", counter / time_taken, step_counter)
            if counter != 0:
                tb_writer.add_scalar("Estimated time remaining (min)", (time_taken / counter)
                                     * max((max_frames - counter), 0) / 60, step_counter)
            next_step += step_size
            step_counter += 1

        # check control queue
        try:
            c = c_q.get_nowait()
        except Empty:
            # nothing in there
            c = None

        # something in control queue -> close video and exit
        if c is not None:
            logger.debug("Ordered to stop")
            cap.close()
            exit()

        if not counter % select:
            img_batch.append(f.to_ndarray(format="rgb24"))
            time_stamps.append(start_time + f.time)

        if len(img_batch) >= batch_size or counter == (max_frames - 1):            
            while 1:
                try:
                    q.put_nowait([img_batch, time_stamps, enc_part])
                    break
                except Full:
                    # check control queue
                    try:
                        c = c_q.get_nowait()
                    except Empty:
                        # nothing in there
                        c = None

                    # something in control queue -> realease video and exit
                    if c is not None:
                        logger.debug("Ordered to stop")
                        cap.close()
                        exit()
                    time.sleep(0.1)

            
            img_batch = []
            time_stamps = []
    # jump here after successfully looking at all frames
    else:
        # make sure remaining frames are processed
        if len(img_batch) > 0:
            while 1:
                try:
                    q.put_nowait([img_batch, time_stamps, enc_part])
                    break
                except Full:
                    # check control queue
                    try:
                        c = c_q.get_nowait()
                    except Empty:
                        # nothing in there
                        c = None

                    # something in control queue -> realease video and exit
                    if c is not None:
                        logger.debug("Ordered to stop")
                        cap.close()
                        exit()
                    time.sleep(0.1)

    logger.debug("Frame limit reached. Stopping")
    cap.close()
    
def to_list(t: np.ndarray) -> list:
    if t is None:
        return None
    else:
        return list(t)


def detect(detector: Detector,
           ht: HomographyTransformer,
           q: Queue,
           c_q: Queue,
           o_q: Queue,
           count_q: Queue):
    """Thread target for a single detector

    Args:
        detector (Detector): Detector to use
        ht (HomographyTransformer): Homography transformer
        q (Queue): Queue to grab images from
        c_q (Queue): Queue to listen to for stop order
        o_q (Queue): Output queue
        count_q (Queue): Queue for counting all processed frames
    """
    logger = logging.getLogger(th.current_thread().name)
    logger.debug("Starting detector thread")
    while 1:
        try:
            img, time_stamps, enc_part = q.get(timeout=0.01)
        except Empty:
            # check control queue
            try:
                c = c_q.get_nowait()
            except Empty:
                # nothing in there
                c = None

            # something in control queue -> exit
            if c is not None:
                logger.debug("Ordered to stop")
                exit()

            continue

        bboxes = detector.detect(img)

        for bbox, t in zip(bboxes, time_stamps):
            count_q.put(True)
            transformed_points = ht.transform(bbox, enc_part)

            if len(transformed_points) > 0:
                d = [t, -1, -1, -1, -1]
                vera_list = [[p, b] for p, b in zip(transformed_points, bbox) if b[5] == "Vera"]
                nanuq_list = [[p, b] for p, b in zip(transformed_points, bbox) if b[5] == "Nanuq"]
                if len(nanuq_list) == 1 and nanuq_list[0][0] is not None:
                    d[1] = nanuq_list[0][0][0] / ENCLOSURE_SHAPE[0]
                    d[2] = nanuq_list[0][0][1] / ENCLOSURE_SHAPE[1]
                if len(vera_list) == 1 and vera_list[0][0] is not None:
                    d[3] = vera_list[0][0][0] / ENCLOSURE_SHAPE[0]
                    d[4] = vera_list[0][0][1] / ENCLOSURE_SHAPE[1]

                tp = [[to_list(t_), *b[4:]] for t_, b in zip(transformed_points, bbox)]
                o_q.put([d, t, bbox, tp, enc_part])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hm",
                        type=str,
                        required=True,
                        help="Homography mapping base directory")
    parser.add_argument("--det_weights",
                        type=str,
                        required=True,
                        help="Path to trained detection weights")
    parser.add_argument("--ident_weights",
                        type=str,
                        default=None,
                        help="Path to trained identification weights")
    parser.add_argument("--videos",
                        type=str,
                        nargs="*",
                        required=True,
                        help="Input video to analyze")
    parser.add_argument("--out",
                        type=str,
                        required=True,
                        help="Directory to write outputs to")
    parser.add_argument("--det_conf",
                        type=float,
                        default=0.6,
                        help="Detection confidence threshold")
    parser.add_argument("--ident_conf",
                        type=float,
                        default=0.5,
                        help="Identification confidence threshold")
    parser.add_argument("--batch",
                        type=int,
                        default=8,
                        help="Batch size to use")
    parser.add_argument("--ident_batch",
                        type=int,
                        default=16,
                        help="Batch size for the identification network")
    parser.add_argument("--one_class",
                        action="store_true",
                        help="Consider only one class for classification")
    parser.add_argument("--max_vid_t",
                        type=int,
                        default=5,
                        help="Maximum number of video threads allowed active at the same time")
    parser.add_argument("--max_det_t",
                        type=int,
                        default=2,
                        help="Maximum number of detector threads allowed active at the same time")
    parser.add_argument("--tb_log",
                        type=str,
                        default=None,
                        help="Path to log to with tensorboard. \
                            If not given, no logging will be performed through tensorboard")
    parser.add_argument("--select",
                        type=int,
                        default=1,
                        help="Forward every <--select>th frame to the neural networks. Must be larger than 0")
    parser.add_argument("--qsize",
                        type=int,
                        default=20,
                        help="Size of queue to send batches over")

    args = parser.parse_args()

    assert args.select > 0, "--select has to be larger than 0"

    logging.basicConfig(level="DEBUG", format=DEFAULT_FORMAT)
    
    if os.path.exists(args.out):
        if not os.path.isdir(args.out):
            raise NotADirectoryError(
                "Output path {} exists, but is not a directory".format(args.out)
                )
    else:
        os.makedirs(args.out, exist_ok=True)

    ht = HomographyTransformer(args.hm)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # image queue
    img_q = Queue(args.qsize)
    # controll queues
    v_c_q = Queue()
    d_c_q = Queue()
    # output queues
    out_q = Queue()
    count_q = Queue()

    video_threads = []

    for i, v in enumerate(args.videos):
        video_threads.append(Thread(target=read_video,
                                    kwargs={
                                        "video_file": v,
                                        "batch_size": args.batch,
                                        "q": img_q,
                                        "c_q": v_c_q,
                                        "tb_log": args.tb_log,
                                        "select": args.select
                                    },
                                    name="Video {}".format(i)))

    det_threads = []

    for i, d in enumerate(range(args.max_det_t)):

        det_threads.append(Thread(target=detect,
                                  kwargs={
                                      "detector": Detector(
                                          args.det_weights,
                                          device,
                                          args.ident_weights,
                                          args.det_conf,
                                          args.ident_conf,
                                          args.one_class,
                                          args.ident_batch),
                                      "ht": ht,
                                      "q": img_q,
                                      "c_q": d_c_q,
                                      "o_q": out_q,
                                      "count_q": count_q},
                                  name="Detector {}".format(i)))

    start_time = time.time()
    try:

        logging.debug("Starting detector threads")

        for d in det_threads:
            d.start()

        logging.debug("Detector threads started")

        logging.debug("Starting video threads")

        # start the maximum nuber of video threads in parallel
        v_t_counter = 0
        # marker for logging of full image queue
        full_logged = False
        while 1:
            if len(th.enumerate()) - (args.max_det_t + 1) < args.max_vid_t and not v_t_counter >= len(video_threads):
                video_threads[v_t_counter].start()
                v_t_counter += 1
                continue

            if not full_logged and img_q.full():
                # Warn for full image queue only once
                logging.warning("Image queue is full")
                full_logged = True

            # check for all video threads to be done
            if all([not v.is_alive() for v in video_threads]):
                break
            time.sleep(0.1)
        logging.debug("Waiting for video threads to finish")
        logging.debug("Video threads finished")

        logging.debug("Ordering detector threads to stop")
        for _ in det_threads:
            d_c_q.put(1)

        for d in det_threads:
            d.join()
        logging.debug("Detector threads stopped")

    except KeyboardInterrupt:
        logging.debug("Ordering to stop everything")
        for _ in video_threads:
            v_c_q.put(1)
        for _ in det_threads:
            d_c_q.put(1)
        for v in video_threads:
            if v.is_alive():
                v.join()
        for d in det_threads:
            d.join()
        logging.debug("Everything stopped")

    end_time = time.time()

    logging.debug("Starting to collect data points")
    enc_points = {}
    pred_bboxes = {}
    pred_tp = {}

    while 1:
        try:
            p, t, b, tp, enc_part = out_q.get_nowait()
        except Empty:
            break
        if enc_part not in enc_points:
            enc_points[enc_part] = []
            pred_bboxes[enc_part] = []
            pred_tp[enc_part] = []
        enc_points[enc_part].append(p)
        pred_bboxes[enc_part].append([t, b])
        pred_tp[enc_part].append([t, tp])

    for enc_part in enc_points.keys():
        enc_points[enc_part].sort(key=lambda x: x[0])
        pred_bboxes[enc_part].sort(key=lambda x: x[0])
        pred_tp[enc_part].sort(key=lambda x: x[0])

    logging.debug("Done collecting data points and sorting")

    for k, v in enc_points.items():
        print("Found {} datapoints for enclosure part {}".format(len(v), k))
    
    print("Mean FPS: {:.2f}".format(count_q.qsize() / (end_time - start_time)))
    
    traj_out = os.path.join(args.out, "traj_{:0>2}.csv")
    bbox_out = os.path.join(args.out, "bbox_{:0>2}.json")
    traj_raw_out = os.path.join(args.out, "traj_raw_{:0>2}.json")
    
    for enc_part in enc_points.keys():
        # TODO collect and write this dataframe in a separate thread
        data = pd.DataFrame(enc_points[enc_part], columns=["time", "x1", "y1", "x2", "y2"])
        data.to_csv(traj_out.format(enc_part), index=False)
        
        # TODO use correct bbox format
        with open(bbox_out.format(enc_part), "w") as f:
            json.dump(pred_bboxes[enc_part], f, indent=1)
            
        with open(traj_raw_out.format(enc_part), "w") as f:
            json.dump(pred_tp[enc_part], f, indent=1)


if __name__ == "__main__":
    main()
