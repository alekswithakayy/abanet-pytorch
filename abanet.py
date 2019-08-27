from util.parser import collect_args

import torch
import train_cam
import make_cam
import make_ir_label
import train_irn
import irn_inference

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    args = collect_args()
    args.cuda = args.num_gpus > 0

    if args.train_cam:
        train_cam.run(args)
    if args.make_cam:
        make_cam.run(args)
    if args.make_ir_label:
        make_ir_label.run(args)
    if args.train_irn:
        train_irn.run(args)
    if args.make_ins_seg_labels:
        pass
    if args.make_sem_seg_labels:
        pass
    if args.irn_inference:
        irn_inference.run(args)
