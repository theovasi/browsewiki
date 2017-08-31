from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField
from wtforms.validators import DataRequired, url

class ScatterGatherForm(FlaskForm):
    cluster_view = RadioField('cluster_view', choices=[(i,
                              'cluster_{}'.format(i)) for i in range(12)])
