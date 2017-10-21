from flask import session
from flask_wtf import FlaskForm
from wtforms import RadioField, SelectMultipleField, widgets
from wtforms.validators import DataRequired, url
import joblib


class ScatterGatherForm(FlaskForm):
    cluster_view = RadioField('cluster_view',
            choices=[])
    cluster_select = SelectMultipleField('cluster_select', 
            choices=[],
            option_widget=widgets.CheckboxInput(),
            widget=widgets.ListWidget(prefix_label=False))
