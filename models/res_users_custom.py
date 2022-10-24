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
 
class user_group_custom(models.Model):
	_name = 'custom.group'
	_description = 'custom.group'
	category_id = fields.Char()