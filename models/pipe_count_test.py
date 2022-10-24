# -*- coding: utf-8 -*-
from collections import defaultdict
from odoo import api, models, fields, tools, _
from odoo.tools.float_utils import float_is_zero
from odoo.exceptions import UserError, ValidationError
import sys
import os
import cv2 as cv
import numpy as np
import pdb
import io
from datetime import date, datetime, timedelta
import base64
import time
import detect_latest
import os
import psutil
# import etree
process = psutil.Process(os.getpid())


class PipeCount(models.Model):
    _name = "pipe.count"
    _description = "pipe.count"
    _rec_name = 'namex' 

    name = fields.Char("Name", default="New")
    state = fields.Selection([("draft", "Draft"), ("confirmed", "Confirmed"), ('counted','Counted')],string='Status', default="draft" )
    image_id = fields.Char(string="Product Image", attachment=False)
    out_put_img = fields.Binary(string="Output Image")
    final_img = fields.Binary(string="Rotated Output Image")
    pipe_count = fields.Float("Count")
    confirmed_date = fields.Datetime()
    speed = fields.Float("Speed")
    templatez = fields.Selection(
        [("t1", "Normal Pipe"), ("t2", "Pipe in Pipe")], string="Template"
    )
    modelz = fields.Selection(
        [("m1", "Model_1"), ("m2", "Model_2"), ("m3", "Model_3"), ("m4", "Model_4"),("m5", "Model_5"),
        ("m6", "Model_6"),("m7", "Model_7"),("m8", "Model_8"),("m9", "Model_9"),("m10", "Model_10")],
        string="Model"
    )
    infer_size = fields.Char(string="Inference_Size", default=1280, readonly=True)
    reference_num = fields.Char(string="Ref_num")

    namex = fields.Char(string="Request Number", readonly=True, copy=False, index=True, default='New')
    product = fields.Char("Product" )
    request_date = fields.Datetime(default=datetime.today())
    status = fields.Char("Status",default='Requested')
    link_id = fields.Many2one('pipe.count',string="Link Img")
    
    # @api.model
    # def fields_view_get(self, view_id=None, view_type='form', toolbar=False, submenu=False):
    #     res = super(PipeCount, self).fields_view_get(view_id=view_id, view_type=view_type, toolbar=toolbar, submenu=submenu)
    #     doc = etree.XML(res['arch'])
    #     print("ZZZZZZZZZZz",self)
    #     print("AAAAAAAAAAAAAA",self._context)
    #     if not self._context.get('uid') == 1:
    #         if view_type == 'form':
    #             nodes = doc.xpath("//form")
    #             for node in nodes:
    #                 node.set('create', "false")
    #             res['arch'] = etree.tostring(doc)                   
    #     return res
    
    @api.model
    def create(self, vals):
        if not vals.get('namex') or vals['namex'] == _('New'):
            seq = self.env['ir.sequence'].browse(self.env.ref('pipi_count.seq_prix_req').id)
            vals['namex'] = seq.next_by_id() or _('New')
            return super(PipeCount, self).create(vals)

    def request(self):
        return {  'namex': _('Pipe Count'),
        'view_mode': 'form',
        'view_id': False,           
        'view_type': 'form', 
        'res_model': 'pipe.count',          
        'type': 'ir.actions.act_window',            
        'target': 'self',
        }
        #self.write({'state':'processing','status':'Processing'})


    def confirm(self):
        if self.state == "draft":
            self.write(
                {
                    "name": self.env["ir.sequence"]
                    .browse(self.env.ref("pipi_count.seq_prix_count").id)
                    .next_by_id(),
                    "state": "confirmed",
                    "confirmed_date": datetime.now(),
                }
            )

    def get_pipe_count(self):
        start_time = time.time()


        # default_file =  r'/home/santhosh/Medium_pipe_final/Input/1.jpg'
        # path_list = default_file.split(os.sep)

        # outimg = path_list[2]
        # nparr = np.frombuffer(base64.b64decode(self.image_id), np.uint8)
        # src = cv.imdecode(nparr, cv.IMREAD_COLOR)
        # import pdb;
        # pdb.set_trace()
        id_1 = self.image_id
        # active_id = self.env.context.get("active_ids", False)
        active_id = self.id
        print("ref id", active_id)

        # customer = self.env['res.partner'].browse(active_id)

        # ref_id = self.env["pipe.count"].browse(active_id)

        # print("ref id", ref_id.ids[0])
        i = 0
        path = "/odoo/Pipex/pipi_count/static/src/img/"
        save_path = path + "orig_" + str(active_id) + ".png"
        src1 = "/odoo/Pipex/pipi_count/static/src/img/g.png"
        nparr = np.frombuffer(base64.b64decode(self.image_id), np.uint8)
        src = cv.imdecode(nparr, cv.IMREAD_COLOR)
        cv.imwrite(save_path, src)
        # src = 'https://ultralytics.com/images/zidane.jpg'
        print("source:", save_path)
        # h, w, c = src.shape
        # print(h,w,c,'****************************')
        #######1,7 and 9 are eliminated
        if self.modelz == "m1":
            weights = "/home/prixgen-gpu/Videos/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_pipe_22_12_2021_s_32b.pb"
        elif self.modelz == "m2":
            weights = "/odoo/Pipex/pipi_count/static/src/tf_mod/1st_normal_yolov5x6.pb"
        elif self.modelz == "m3":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5s_32.pb"
            )
        elif self.modelz == "m4":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5s.pb"
            )
        elif self.modelz == "m5":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5s6.pb"
            )
        elif self.modelz == "m6":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5s6_32.pb"
            )
        elif self.modelz == "m7":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5s_backgroundmix_16.pb"
            )

        elif self.modelz == "m8":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5x6.pb"
            )           
        elif self.modelz == "m9":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5l_backgroundmix_8.pb"
            )
        elif self.modelz == "m10":
            weights = (
                "/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_in_pipe_yolov5x6_background_mix_6.pb"
            )

        # Check if image is loaded fine##
        if src is None:
            # print ('Error opening image!')
            # print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
            return -1

        # modeld="/odoo/Pipex/pipi_count/models/pipe_in_pipe_yolov5s.pt"
        ####  for normal pipe
        if self.templatez == "t1":
            ####here put weights file path for  pipe in pipe
            weights="/home/prixgen-gpu/Videos/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_pipe_22_12_2021_s_32b.pb"
            print("selected Normal Pipe")
            r, img = detect_latest.run(
                source=save_path,
                templates="t1",
                imgsz=[1280 ,1280],
                conf_thres=0.35,
                iou_thres=0.4,
                hide_labels=True,
                hide_conf=True,
                agnostic_nms=True,
                weights=weights,
                nosave=False,
            )

            is_success, buffer = cv.imencode(".png", img)
            io_buf = io.BytesIO(buffer)
            img_data = base64.b64encode(io_buf.getvalue())
            # print("after read image is: ",img_data)
            end_time = time.time()
            self.write(
                {
                    "pipe_count": r,
                    "out_put_img": img_data,
                    "speed": end_time - start_time,
                }
            )
        ####  for pipe in pipe
        if self.templatez == "t2":
            ####here put weights file path for  pipe in pipe
            # weights='/odoo/Pipex/pipi_count/static/src/tf_mod/best.pb'
            weights="/home/prixgen-gpu/Videos/odoo/Pipex/pipi_count/static/src/tf_mod/pipe_pipe_22_12_2021_s_32b.pb"
            #print("selected Pipe in Pipe")
            r, img = detect_latest.run(
                source=save_path,
                templates="t2",
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
            is_success, buffer = cv.imencode(".png", img)
            io_buf = io.BytesIO(buffer)
            img_data = base64.b64encode(io_buf.getvalue())
            # print("after read image is: ",img_data)
            end_time = time.time()
            self.write(
                {
                    "pipe_count": r,
                    "out_put_img": img_data,
                    "speed": end_time - start_time,
                }
            )
            print("memory usage: ",process.memory_percent())
            print('Percentage of used RAM :',psutil.virtual_memory().percent,'%')
       
        self.out_put_img = img_data
        self.state = 'counted'

        # return {
        #         'name': 'Pipex_count_form',
        #         'type': 'ir.actions.act_window',
        #         'res_model': 'pipe.count',
        #         'view_mode': 'form',
        #         'view_type': 'form',
        #         # 'views': [(pipx_count_formz, 'form')],
        #         'view_id': self.env.ref("pipi_count.pipx_count_formz").id,
        #         'target': 'current',
        #         'domain':[('pipe_count','=',r),('out_put_img','=',img_data),('image_id','=',nparr),('out_put_img','=',False)],
        #         'context':{'default_pipe_count':self.pipe_count,'default_out_put_img':self.out_put_img,'default_image_id':self.image_id}
        #         }


class Pipe_Request(models.Model):
    _name = "pipe.request"
    _description = "Pipe Request Screen"

    request_no = fields.Char("Request Number", default="New")
    product = fields.Char("Product")
    request_date = fields.Datetime()
    status = fields.Char("Status")
    state = fields.Selection(
        [("draft", "Draft"), ("confirmed", "Confirmed")], default="draft"
    )
