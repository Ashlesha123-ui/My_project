import copy
import json
import io
import logging
import lxml.html
import datetime
import ast
from odoo.http import request
from collections import defaultdict
from math import copysign

from dateutil.relativedelta import relativedelta

from odoo.tools.misc import xlsxwriter
from odoo import models, fields, api, _
from odoo.tools import config, date_utils, get_lang
from odoo.osv import expression
from babel.dates import get_quarter_names, parse_date
from odoo.tools.misc import formatLang, format_date
from odoo.addons.web.controllers.main import clean_action
from odoo.tools.safe_eval import safe_eval
from ast import literal_eval

_logger = logging.getLogger(__name__)


class PrixCountThings(models.AbstractModel):
    _name = "prix.count.things"
    _description = 'Count Things'


    def get_dashboard_information(self):
        uid = request.session.uid

        if uid == 2:
            domain = []
        else:
            domain = [('create_uid','=',uid)]
        counts = self.env['pipe.count'].search(domain)
        tot_speed = sum(counts.mapped('speed'))
        if len(counts) > 0:
        	avg_speed = tot_speed/len(counts)
        else:
        	avg_speed = 0

        info = {'tot_counts': sum(counts.mapped('pipe_count')),'avg_speed':round(avg_speed,3),}
        return info


    def prix_count_action(self):
        # return {
        #     'name': _('Process Count'),
        #     'type': 'ir.actions.client', 
        #     'tag': 'prix_counting_process',
        #     'target': 'current',
        #     }
        action = self.env["ir.actions.actions"]._for_xml_id("pipi_count.action_pip_count")
        return action
    

    def count_history(self):
        uid = request.session.uid

        if uid == 2:
            domain = []
        else:
            domain = [('create_uid','=',uid)]

        return {
            'name': _('Count History'),
            'type': 'ir.actions.act_window',
            'view_mode': 'tree,form',
            'res_model': 'pipe.count',
            'context': {'create': False, 'edit':False,'delete': False},
            'domain': domain,
            'views': [[False, 'list'],[False, 'form']],
            'target': 'current',
        }

    def count_request(self):
        uid = request.session.uid

        if uid == 2:
            domain = []
        else:
            domain = [('create_uid','=',uid)]

        return {
            'name': _('Request'),
            'type': 'ir.actions.act_window',
            'view_mode': 'tree,form',
            'res_model': 'pipe.count',
            'domain': domain,
            'views': [[False, 'list'],[False, 'form']],
            'target': 'current',
        }
    def create_user(self):
        uid = request.session.uid

        if uid == 2:
            domain = []
        else:
            domain = [('create_uid','=',uid)]

        return {
            'name': _('Users'),
            'type': 'ir.actions.act_window',
            'view_mode': 'tree,form',
            'res_model': 'res.users',
            'domain': domain,
            'views': [[False, 'list'],[False, 'form']],
            'target': 'current',
        }

class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    def open_template_user(self):
        action = self.env["ir.actions.actions"]._for_xml_id("base.action_res_users")
        action['res_id'] = literal_eval(self.env['ir.config_parameter'].sudo().get_param('base.template_portal_user_id', 'False'))
        action['views'] = [[self.env.ref('base.view_users_form').id, 'form']]
        return action
