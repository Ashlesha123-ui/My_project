# -*- coding: utf-8 -*-
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
import cv2 as cv
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

class ActiveSession(models.Model):

    _name = 'active.session'

    user_id = fields.Char(string='user')
    session_active = fields.Char(string='session active')


class ResUserInherit(models.Model):

    _inherit = 'res.users'

    # login_restrict = fields.Boolean(string='login restrict')
    # cookie = fields.Char(string='cookie')
    session_active = fields.Char(string='session active')

class Extension_Home(Home):

    @http.route()
    def web_login(self, redirect=None, **kw):
        # ip_address = request.httprequest.environ['REMOTE_ADDR']
        # print('ip_address----------------------',ip_address)
        res={}
        #Inherit Login controller and redirect to Custom Page
        if 'login' in kw:
            #your_logic_goes_here()

            request.params['login_success'] = False
            if request.httprequest.method == 'GET' and redirect and request.session.uid:
                return http.redirect_with_hash(redirect)

            if not request.uid:
                request.uid = odoo.SUPERUSER_ID

            values = request.params.copy()
            try:
                values['databases'] = http.db_list()
            except odoo.exceptions.AccessDenied:
                values['databases'] = None
            # pdb.set_trace()
            # user = request.env['res.users'].sudo().search([('id', '=', request.session.uid)])
            # print('user----------------------',user)


            user_id = request.env['res.users'].sudo().search([('login', '=', request.params['login'])])
            # session_data = request.env['active.session'].sudo().search([])

            print('active_session---------------',user_id.session_active)
            if user_id.session_active != ' ':  
                print("User already logged in. Log out from " \
                                              "other devices and try again.")
                if 'error' in request.params and request.params.get('error') == 'access':
                        values['error'] = _('User already logged in. Log out from " \
                                              "other devices and try again.')
                # url = "/restrict"
                # return werkzeug.utils.redirect(url)
                res = {}
                res = {
                'token' : user_id.session_active
                }
                return http.request.render('lecca.login-restrict', res)
            else:
                # uid = request.session.authenticate(request.session.db, request.params['login'], request.params['password'])            
                if request.httprequest.method == 'POST':
                    old_uid = request.uid
                    
                    # print('user=============',request.session.uid)
                    try:
                        print('in tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                        uid = request.session.authenticate(request.session.db, request.params['login'], request.params['password'])
                        request.params['login_success'] = True
                        print('login_restrict---------------',request.session)
                        url = '/web/True'
                        var = random.randint(0,999999999999999999)
                        user = request.env['res.users'].search([('id', '=', uid)])
                        user.sudo().write(
                            {
                                'session_active':request.session['session_token'],
                                # "cookie":var
                            })
                        request.session['uid'] = uid
                        # active_session_create = {
                        #     'session_active':request.session,
                        #     'user_id':request.params['login'],
                        # }

                        # # Creates Leeca Count
                        # active_session = request.env['active.session'].sudo().create(active_session_create)
                        print('request.session============',request.session['uid'])
                        # request.set_cookie('session_id', httprequest.session.sid, max_age=90 * 24 * 60 * 60, httponly=True)
                        # res = {}
                        # res = {
                        # 'token' : user_id.session_active
                        # }
                        return werkzeug.utils.redirect(url)

                    except odoo.exceptions.AccessDenied as e:
                        print('in exceptttttttttttttttttttttttttt')
                        request.uid = old_uid
                        if e.args == odoo.exceptions.AccessDenied().args:
                            values['error'] = _("Wrong login/password")
                        else:
                            values['error'] = e.args[0]
                    

                else:
                    if 'error' in request.params and request.params.get('error') == 'access':
                        values['error'] = _('Only employees can access this database. Please contact the administrator.')

                if 'login' not in values and request.session.get('auth_login'):
                    values['login'] = request.session.get('auth_login')

                if not odoo.tools.config['list_db']:
                    values['disable_database_manager'] = True

                response = request.render('web.login', values)
                # response.set_cookie('sid', session.sid, max_age=1*60, expires=int(time.time())+1*60, httponly=True)
                response.headers['X-Frame-Options'] = 'DENY'
                # s['username'] = uid
                # Pass Details to webpage

            # res = {
            #     'uid': uid,
            #     'cookie': var,
            # }
            # print("jghggggggggggggggggggggggggggggggggggggggggggggggggggffffffff",res)
        return super(Extension_Home, self).web_login()


class Session_Inherit(Session):


    @http.route('/web/session/logout', type='http', auth="public", website=True)
    def logout(self, redirect='/web',**kw):
        ip = request.httprequest.environ['REMOTE_ADDR']
        print('i came here---------------',ip)
        user = request.env['res.users'].sudo().search([('id', '=', request.session.uid)])
        print('lllllllllllllllllllllllaaaaaaaaaaaaaaaaaaaaakkkkkkkkkkkkkksssssshhhhhhmmiiiii---------------',user)
        # session_data = request.env['active.session'].sudo().search([])
        # session.sudo().unlink()
        # print('login_restrict---------------',request.session)
        # print('active_session---------------',session_data)
        # if user.cookie:
        user.sudo().write(
        {
            "session_active": ' ',
        })
        # request.browser.delete_cookie('session_id', domain=HOST)
        request.session.logout(keep_db=True)
        return werkzeug.utils.redirect(redirect, 303)

        # else:
        #     request.session.logout(keep_db=True)
        #     return werkzeug.utils.redirect(redirect, 303)


    @http.route('/logout/all', type='http', auth="none")
    def logout_all(self, redirect='/web', f_uid=False):
        ip = request.httprequest.environ['REMOTE_ADDR']
        # print('bgggggggggghhhhhhhhhhhghytttttttttttttttujjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj---------------',uid)
        user = request.env['res.users'].sudo().search([('session_active', '=', f_uid )])
        print('bgggggggggghhhhhhhhhhhghytttttttttttttttujjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj---------------',user)
        # session_data = request.env['active.session'].sudo().search([])
        # print('login_restrict---------------',request.session)
        # print('active_session---------------',session_data)

        user.sudo().write(
        {
            "session_active": ' ',
        })
       
        request.session.logout(keep_db=True)
        print('loggedddddddddd outttttttttttttttttt')
        return werkzeug.utils.redirect(redirect, 303)


class LeccaCount(models.Model):
    _name = 'lecca.count'
    _rec_name = 'product'

    product = fields.Char("Product")
    attachment = fields.Binary(string='Image')
    # count = fields.Float(string='Count')
    template_id = fields.Many2one('lecca.count.template')

    image_id = fields.Char(string="Product Image", attachment=False)
    out_put_img = fields.Binary(string="Output Image")
    final_img = fields.Binary(string="Rotated Output Image")
    pipe_count = fields.Integer("Count")
    confirmed_date = fields.Datetime()
    speed = fields.Float("Speed")

class leccaCountTemplate(models.Model):
    _name = 'lecca.count.template'
    _rec_name = 'name'

    name = fields.Char("Name",store=True)
    template_image = fields.Binary(string='Image')    

class Attachment(models.Model):
    _inherit = 'ir.attachment'

    attach_rel = fields.Many2many('lecca.count', 'attachment', 'attachment_id3', 'document_id',string="Attachment", invisible=1 )

class lecca(http.Controller):

    _inherit = 'lecca.count'

    # Template Page
    @http.route('/web/True', type='http', auth='user', website=True)
    def show_custom_webpage(self, **kw):
        # Check Login
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)
        else:
            # Redirect to Templates Page & Pass Details to webpage
            res = {}
            t = request.env['lecca.count.template'].search([])
            res = {
                'template': t,
            }
            return http.request.render('lecca.index', res)


    # Lecca Count Page            
    @http.route('/Form', type='http', auth='user', website=True)
    def show_request_form(self, **kw):
        print('kw-----------------',kw)
        if not request.session.uid:
            return http.request.render('lecca.login-user', {})
        else:
            res = {}     
            lecca_count_template_id = kw.get('id')

            # Get template Details
            t = request.env['lecca.count.template'].search([('id', '=', lecca_count_template_id)])

            template_id = t.id
            lecca_count_create = {
                'template_id':kw.get('id'),
            }

            # Creates Leeca Count
            lecca_count = request.env['lecca.count'].create(lecca_count_create)

            # Pass Details to webpage
            res = {
                'lecca_count_template': t,
                'lecca_count': lecca_count,
                'lecca_count_template_id': template_id,
            }
            return http.request.render('lecca.request_form', res)

    # Lecca Count Details Page
    @http.route('/Details', type='http', auth='user', website=True)
    def update_record(self, **kw):  
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)
        else:
            # Update the current created lecca record
            lecca_count_template_id = kw.get('id')
            count_id = kw.get('count_id')
            print('lecca_count_template /Details-------------------------',lecca_count_template_id)

            lecca_template = request.env['lecca.count.template'].search([('id', '=', lecca_count_template_id)])
            templatez = lecca_template.name
            print("template name------------",templatez)

            # Count Algorithm
            if(templatez):
                res = {}
                ran_number = random.random()
                img = kw.get('image')
                #print("i came here@@@@@@@@@@",img)
                temp = io.BytesIO()
                #img.convert('RGB')
                img.save(temp)

                count_details = {
                    'product':kw.get('product'),
                    'attachment': base64.b64encode(temp.getvalue()),
                }
                lecca_count = request.env['lecca.count'].search([('id', '=', count_id)])
                lecca_count.write(count_details)
                start_time = time.time()
                input_img = lecca_count.attachment

                path = "/home/prixgen-gpu/Videos/odoo/custom/lecca/static/img/"
                save_path = path + "orig_" + str(ran_number) + ".png"
                nparr = np.frombuffer(base64.b64decode(input_img), np.uint8)
                src = cv.imdecode(nparr, cv.IMREAD_COLOR)
                cv.imwrite(save_path, src)


                # Check if image is loaded fine##
                if src is None:
                    print ('Error opening image!')
                    # print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
                    return -1

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

                    is_success, buffer = cv.imencode(".png", img)
                    io_buf = io.BytesIO(buffer)
                    img_data = base64.b64encode(io_buf.getvalue())
                    # print("after read image is: ",img_data)
                    end_time = time.time()
                    lecca_count.write(
                        {
                            "pipe_count": r,
                            "out_put_img": img_data,
                            "speed": end_time - start_time,
                        }
                    )
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
                    is_success, buffer = cv.imencode(".png", img)
                    io_buf = io.BytesIO(buffer)
                    img_data = base64.b64encode(io_buf.getvalue())
                    # print("after read image is: ",img_data)
                    end_time = time.time()
                    lecca_count.write(
                        {
                            "pipe_count": r,
                            "out_put_img": img_data,
                            "speed": end_time - start_time,
                        }
                    )

                res = {
                    'count1': lecca_count,
                    'lecca_count_template_id': lecca_count_template_id,
                }
                return request.render("lecca.count-details", res)

            else:
                return request.render("lecca.index", {})

    # History Page
    @http.route('/History', auth='user', website=True)
    def show_history(self, **kw):

        # Check Login
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)

        # Redirect to History Page
        else:
            res = {}
            products = request.env['lecca.count'].search([('create_uid','=',request.uid),('product', '!=', False)])
            res = {
                'count': products,
            }
            return http.request.render('lecca.history', res)


    # Image Preview
    @http.route(['/image-preview'], type='http', auth="public", website=True, sitemap=False)
    def image_preview(self, **post):
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)
        else:
            lecca_id = post.get('id')
            res = {}
            lecca_count = request.env['lecca.count'].search([('id', '=', lecca_id)])    
            res = {
                'lecca_count': lecca_count,
            }
            return http.request.render('lecca.image-preview',res)


    # Print Report
    @http.route('/print-report/<id>', methods=['POST', 'GET'], csrf=False, type='http', auth="user", website=True)
    def print_report(self,id, **kw):
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)
        else:
            # Get the id of record
            lecca = request.env['lecca.count'].search([('id', '=', id)], limit=1)
            pdf_ = request.env.ref('lecca.count_report').sudo()._render_qweb_pdf([lecca.id])[0]
            pdfhttpheaders = [
            ('Content-Type', 'application/pdf'),
            ('Content-Length', len(pdf_)),
            ]
            return request.make_response(pdf_, headers=pdfhttpheaders)

        
    # How To Page
    @http.route('/How-to', type='http', auth='user', website=True)
    def how_to(self, **kw):
        user = request.env['res.users'].sudo().search([('id', '=',request.session['uid'])])
        print('userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',request.session['session_token'])
        # if not request.session.uid:
        if user.session_active == ' ': 
            request.session.logout(keep_db=True)
            # return http.request.render('lecca.login-user', {})
            redirect='/web'
            return werkzeug.utils.redirect(redirect, 303)
        else:  
            return http.request.render('lecca.how-to',{})

    # Login Restrict Page
    @http.route('/restrict', type='http', auth='user', website=True)
    def login_restrict(self, **kw):  
        return http.request.render('lecca.login-restrict',{})

