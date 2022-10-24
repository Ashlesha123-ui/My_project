"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
import pyshine as ps

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from experimental import attempt_load
from datasets import LoadStreams, LoadImages
from general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from plots import colors, plot_one_box
from torch_utils import select_device, load_classifier, time_sync
from PIL import Image, ImageFont, ImageDraw, ImageOps

def sort_bbox(bbox, method="left-to-right"):

    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = sorted(bbox, key=lambda b: b[3], reverse=reverse)
    # return the list of sorted contours and bounding boxes
    return boundingBoxes

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=True,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    print("1st stage passed!!!!!")
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    print("2st stage passed!!!!!")
    print("weights:",weights)
    #weights="/odoo/Pipex/pipi_count/models/pipe_in_pipe_yolov5s.pt"
    if pt:
        print("inside loop")
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', "custom", force_reload=True,autoshape=True)
        ###model.load_state_dict(torch.load(weights)['model'].state_dict())
        model = attempt_load(weights, map_location=device,inplace=True, fuse=True)  # load FP32 model
        print("names",model)
        print()
        print()
        print()
        print()
        #stride = int(model.stride.max())  # model stride
        #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("3rd stage passed!!!!!")
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # if pt and device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            print(img)
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            print("img:    ",img)
            #img1 = Image.open(source)  # PIL image
            img1 = cv2.imread(source)[..., ::-1] 
            pred = model(img1,size=1280)
            print('predictions:',pred)
        elif onnx:
            #img1 = cv2.imread(source)[..., ::-1] 
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        pipe_count=0
        pipe_list=[]
        boxes_list=[]
        print("4th stage passed!!!!!")
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            imnot=im0.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        lab=f'{names[int(cls)]}'
                        cord,rrrr=plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        boxes_list.append(cord)
                        if lab=='pipe':
                          #print("ppppppp:",countttt)
                          pipe_count+=1
                        else:
                          continue
                        pipe_list.append(pipe_count)                       
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            #print("box list: ",boxes_list)
            print("5th stage passed!!!!!")
            font = cv2.FONT_HERSHEY_SIMPLEX
            sort_l = sort_bbox(boxes_list, method="top-to-bottom")
            ccc = 1
            for box in sort_l:
                # print('sorted list:',box)

                fnt = ImageFont.truetype(
                    r"./.local/lib/python3.6/site-packages/cv2/qt/fonts/DejaVuSans-ExtraLight.ttf",
                    40,
                    encoding="unic",
                )
                #im0 = Image.fromarray(im0)
                #draw = ImageDraw.Draw(Image.fromarray(im0))
                #print("i came here tooooo!!!!!!")
                text = str(ccc)
                #w, h = draw.textsize(text, font=fnt)
                TEXT_THICKNESS = 2

                # w, h = fnt.getsize(text)
                x, y, w, h = box
                #w, s, e, n = wms.contents[list(wms.contents)[0]].boundingBoxWGS84
                # get the box width 
                width = max(x, w) - min(x, w)
                # get the box height 
                height = max(y, h) - min(y, h)
                # compute the center
                center = round(min(y, h)+height/2, 4), round(min(x, w)+width/2, 4)
                # new_x = (x2 - x1 - w) / 2 + x1
                # new_y = (y2 - y1 - h) / 2 + y1
                #center = (x+w//2, y+h//2)
                #center_coordinates = x + w // 2, y + h // 2
                #center_coordinates = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                print("center_coordinates: ",center)
                # print('box',box)
                TEXT_SCALE = 0.5
                TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
                #text_origin = (int(center_coordinates[0]) - text_size[0] / 2, int(center_coordinates[1]) + text_size[1] / 2)
                #print("center_coordinates: ",text_origin)
                #cv2.putText(im0, text, text_origin, font, 1.5, (0,0,0), int(TEXT_THICKNESS), cv2.LINE_AA)
                #cv2.putText(im0, text, (int(center[0]),int(center[1])), TEXT_FACE, TEXT_SCALE, (127,255,127), int(TEXT_THICKNESS), cv2.LINE_AA)
                # print('print what the hell is temp: ',temp)
                # print('count:',ccc)
                # if template_name == "Normal Pipe":
                #draw.text(((new_x, new_y)), text, fill="black", font=fnt)
                # draw.text(((new_x, new_y)),text , fill="black",font = fnt)
                ccc += 1
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            #print("S:  ",pipe_list)
            # font_family = "arial.ttf"
            pipe_len_text = "counts:" + str(len(pipe_list))
            #draw = ImageDraw.Draw(Image.fromarray(im0))
            # W, H = img.size
            # # W = (img.shape[1] - textsize[0]) / 2
            # # H = (img.shape[0] + textsize[1]) / 2
            # t_x = W / 2
            # t_y = H / 2
            img_r = 0
            # font_size = find_font_size(pipe_len_text, draw, width_ratio)
            # font = ImageFont.truetype(font_family, font_size)
            # print(f"Font size found = {font_size} - Target ratio = {width_ratio} - Measured ratio = {get_text_size(text, image, font)[0] / image.width}")
            #w, h = draw.textsize(pipe_len_text, font=fnt)
            # draw.text((0,0), pipe_len_text,fill="black", font=fnt)

            # print("end....................................................")
            #img0 = np.asarray(img)
            # # cv2.putText(img, "No. of pipe counts:"+str(len(pipe_list)), (10, 35), font, 0.7, (255,0,0),2)
            # buffered = BytesIO()
            # img = Image.fromarray(img)
            img = ps.putBText(
                im0,
                pipe_len_text,
                text_offset_x=20,
                text_offset_y=20,
                vspace=10,
                hspace=10,
                font_scale=2.0,
                background_RGB=(20, 210, 4),
                text_RGB=(255, 250, 250),
            )           
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # if (len(pipe_list) == 0):
            #   cv2.putText(im0, "...", (10, 35), font, 1.2, (0,0,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
            # else:
            #   cv2.putText(im0, "No. of pipe counts:"+str(len(pipe_list)), (10, 35), font, 0.8, (255,0,0),2,cv2.FONT_HERSHEY_SIMPLEX)
            #   ####till here#######
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
