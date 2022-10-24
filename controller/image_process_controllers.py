from odoo import models, fields, api
from odoo import http
import random
from odoo.http import request
from odoo.addons.website_sale.controllers.main import WebsiteSale
from odoo.addons.web.controllers.main import Home
from odoo.addons.website.controllers.main import Website
import odoo
import base64
import io
from collections import defaultdict
from odoo import api, models, fields, tools, _
from odoo.tools.float_utils import float_is_zero
from odoo.exceptions import UserError, ValidationError
import sys
import os
import cv2 
import numpy as np
import pdb
from datetime import date, datetime, timedelta
import time
import psutil
from ..models import detect_latest
from odoo.addons.website.controllers.main import Website
from PyPDF2 import  PdfFileReader, PdfFileWriter
from reportlab.pdfgen import canvas
import odoo.addons.web.controllers.main as main
from odoo.addons.web.controllers.main import Home
from odoo.addons.web.controllers.main import Session
import werkzeug.security
# import requests module
from py_session import py_session

global res

class ImageProcessController(http.Controller):

    @http.route('/pipe/img',type='json',auth="public",csrf=False,method=["POST"])
    def ProcessImages(self):
        global res
        resp = request.jsonrequest
        #resp = kw
        print(type(resp.get('image')))

        nparr = np.frombuffer(base64.b64decode(resp.get('image')), np.uint8)
        print(nparr,"yesss you have sent image just now...!!!!!!!!!!")
        ran_number = random.random()
        src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        path = "/home/prixgen-gpu/Videos/odoo/custom/lecca/static/img/"
        save_path = path + "orig_" + str(ran_number) + ".png"
        cv2.imwrite(save_path, src)


        # Check if image is loaded fine##
        if src is None:
            print ('Error opening image!')
            # print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
            return -1
        templatez = "Telescopic Arrangement"
        if templatez == "Sequential Arrangement":
            ####here put weights file path for  pipe in pipe
            #weights="/home/prixgen-gpu/Videos/odoo/custom/lecca/static/pbs/pipe_in_pipe_yolov5s_32.pb"
            weights="/home/prixgen-gpu/Desktop/PIPE_COUNT_APP_ORIGINAL_CURRENTLY_RUNNING/odoo/custom/lecca/static/pbs/best.pb"
            print("selected Normal Pipe")
            r, img = detect_latest.run(
                source=save_path,
                templates="Sequential Arrangement",
                imgsz=[1280 ,1280],
                conf_thres=0.35,
                iou_thres=0.4,
                hide_labels=True,
                hide_conf=True,
                agnostic_nms=True,
                weights=weights,
                nosave=False,
            )

            is_success, buffer = cv2.imencode(".png", img)
            io_buf = io.BytesIO(buffer)
            img_data = base64.b64encode(io_buf.getvalue())
            # print("after read image is: ",img_data)
            end_time = time.time()
            # lecca_count.write(
            #     {
            #         "pipe_count": r,
            #         "out_put_img": img_data,
            #         "speed": end_time - start_time,
            #     }
            # )
        ####  for pipe in pipe
        if templatez == "Telescopic Arrangement":
            ####here put weights file path for  pipe in pipe
            # weights='/odoo/Pipex/pipi_count/static/src/tf_mod/best.pb'
            #weights="/home/prixgen-gpu/Videos/odoo/custom/lecca/static/pbs/pipe_in_pipe_yolov5s_32.pb"
            weights="/home/prixgen-gpu/Desktop/PIPE_COUNT_APP_ORIGINAL_CURRENTLY_RUNNING/odoo/custom/lecca/static/pbs/best.pb"
            #print("selected Pipe in Pipe")
            r, img = detect_latest.run(
                source=save_path,
                templates="Telescopic Arrangement",
                imgsz=[1280 ,1280],
                conf_thres=0.35,
                iou_thres=0.4,
                hide_labels=True,
                hide_conf=True,
                agnostic_nms=True,
                weights=weights,
                nosave=False,
            )
            # print("pipe_cont is: ",r)
            # print("image is: ",img)
            is_success, buffer = cv2.imencode(".png", img)
            io_buf = io.BytesIO(buffer)
            img_data = base64.b64encode(io_buf.getvalue())
        
        
        #r = str({'success':True,"pipe_count": r,"out_put_img": img_data,"status_code": 200 })
        #print(r,"rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        # return {'success':True,"pipe_count": r,"out_put_img": img_data,"status_code": 200 }


    #########################################
        global res
        is_success, buffer = cv2.imencode(".png", src)
        io_buf = io.BytesIO(buffer)
        img_org = base64.b64encode(io_buf.getvalue())
        res = {"out_put_img": img_org}
        #print("ressssssssssssssssssssssssssssss",img_org)
        # return request.render("lecca.request_form", res)

        redirect='/Form/%s' % img_org
        return werkzeug.utils.redirect(redirect)

    # @http.route('/flask-result', type="http", auth='public', website=True)
    # def flask_result(self, **kw):
    #     global res
    #     # print('res---------------------',res)

    #     return request.render("lecca.flask-result",{})

    ##################################