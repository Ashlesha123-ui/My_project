# -*- coding: utf-8 -*-
from collections import defaultdict
from odoo import api, models, fields,tools, _
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


class PipeCount(models.Model):
	_name = 'pipe.count'
	_description ='pipe.count'

	name = fields.Char("Name", default="New")
	state = fields.Selection([('draft','Draft'),('confirmed','Confirmed')],default='draft')
	image_id = fields.Binary(string='Product Image', attachment=False)
	out_put_img = fields.Binary(string='Output Image')
	final_img = fields.Binary(string='Rotated Output Image')
	pipe_count = fields.Float("Count")
	confirmed_date = fields.Datetime()
	speed = fields.Float("Speed")
	templatez = fields.Selection([('t1', 'T1'),('t2', 'T2')],string="Template")
	modelz = fields.Selection([('m1', 'M1'),('m2', 'M2'),('m3','M3'),('m4','M4')],string="Model")
	infer_size = fields.Char(string="Inference_Size", default=1280, readonly=True)

	def confirm(self):
		if self.state == 'draft':
			self.write({'name': self.env['ir.sequence'].browse(self.env.ref('pipi_count.seq_prix_count').id).next_by_id(),
				'state':'confirmed',
				'confirmed_date':datetime.now(),
				})

		# if self.state == 'confirmed':
		# 	print('Yessssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
			

	def get_pipe_count(self):
		start_time = time.time()

		# default_file =  r'/home/santhosh/Medium_pipe_final/Input/1.jpg'
		# path_list = default_file.split(os.sep)
		
		# outimg = path_list[2]
		nparr = np.frombuffer(base64.b64decode(self.image_id), np.uint8)
		src = cv.imdecode(nparr, cv.IMREAD_COLOR)

		# h, w, c = src.shape
		# print(h,w,c,'****************************')


		# Check if image is loaded fine##
		if src is None:
			# print ('Error opening image!')
			# print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
			return -1
			   
		## [load]##
		## [convert_to_gray]##
		## [Convert it to gray]##
		gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
		## [convert_to_gray]##
		## [reduce_noise]##
		## Reduce the noise to avoid false circle detection##
		gray = cv.GaussianBlur(gray,(5,5),0)
		cv.BORDER_DEFAULT
		gray = cv.medianBlur(gray, 5)
		gray = cv.bilateralFilter(gray,5,5,5)
		th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
		                cv.THRESH_BINARY,11, 2)
		rows = gray.shape[0]
		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows /100,
		                           param1=100, param2=30,
		                           minRadius=1, maxRadius=30)
		
		if circles is not None:
		   circles = np.uint16(np.around(circles))
		   number=1
		   font = cv.FONT_HERSHEY_SIMPLEX
		   height= 40
		   
		for cont, i in enumerate(circles, start=1):
		  circles = np.round(circles[0,:]).astype("int")
		  circles = sorted(circles, key=lambda v: [v[0], v[1]])

		  NUM_ROWS = 1000
		  sorted_cols = []
		for k in range(0, len(circles), NUM_ROWS):
		    col = circles[k:k+NUM_ROWS]
		    sorted_cols.extend(sorted(col, key=lambda v: v[1]))
		    circles = sorted_cols  

		for i in circles[0: ]:
		    x, y= i[0], i[1]
		    number=str(number)
		    numbersize= cv.getTextSize(number, font, 1, 2)[0]


		    radius= i[2]
		    cv.circle(src, (x, y), radius, (255, 0, 255), 3)
		    cv.rectangle(src, (x - 12, y - 12), (x + 12, y + 12), (0, 128, 255), -1)   
		    
		    cv.putText(src, number, (x - 5, y + 5) , font, 0.5,(255, 255, 255))
		    
		    number=int(number)
		    number+=1

		r = number-1
		# h, w, c = src.shape
		# print(h,w,c,'****************************')       
		is_success, buffer = cv.imencode('.png',src)
		io_buf = io.BytesIO(buffer)
		img_data = base64.b64encode(io_buf.getvalue())

		
		# output_img= base64.b64encode(nnnnn)
		# attachment = self.env['ir.attachment'].create({
		# 'name': 'output.png',
		# 'type': 'binary',
		# 'datas': otpt_image,
		# 'res_model': 'pipe.count',
		# 'res_id': self.id,
		# 'res_field': 'out_put_img',
		# 'mimetype': 'image/png',
		# })
		end_time = time.time()
		self.write({'pipe_count': r,'out_put_img':img_data,'speed':end_time-start_time})


		return 0

	def rotate(self):
		# imgs = cv.imread('out_put_img')
		# print(imgs)
		# img_flip_lr = cv.flip(imgs, 1)
		# print(img_flip_lr)

		# self.out_put_img=img_flip_lr
		img = cv.imread('/odoo14/Pipe_Count/pipi_count/models/12.jpg')
		print(type(img))
		# print(img.shape)
		print(img.shape)
		img_rotate_90_clockwise = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
		print('rrrrrrrrrrr',img_rotate_90_clockwise)
		self.final_img=img_rotate_90_clockwise
		# self.final_img=np.frombuffer(base64.b64decode(self.out_put_img), np.uint8)
		return 0
		# cv.imwrite('out_put_img', img_flip_lr)

class Pipe_Request(models.Model):
	_name = 'pipe.request'
	_description = 'Pipe Request Screen'

	request_no = fields.Char("Request Number", default="New")
	product = fields.Char("Product")
	request_date = fields.Datetime()
	status = fields.Char("Status")
	state = fields.Selection([('draft','Draft'),('confirmed','Confirmed')],default='draft')


