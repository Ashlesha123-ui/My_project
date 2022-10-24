# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.
{
    'name': 'lecca Website',
    'version': '14.0.6',
    'category': 'Productivity',
    'sequence': 3,
    'summary': 'lecca Website',
    'description': """
    """,
    'website': 'https://www.prixgen.com',
    'depends': ['website','web','base'],

    'data': [
        'views/views.xml',
        'views/sign_in.xml',
        'views/templates.xml',
        'views/login.xml',
        'views/count-entry.xml',
        'views/count-details.xml',
        'views/history.xml',
        'views/menu.xml',
        'views/footer.xml',
        'views/image-preview.xml',
        'security/ir.model.access.csv',
        'views/count_report.xml',
        'views/how-to.xml',
        'views/login_restrict.xml',
        'views/flask_result.xml',
    ],
     
    'installable': True,
    'auto_install': False,
    'application': True,

}
