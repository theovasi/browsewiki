from flask_wtf import FlaskForm
from wtforms import RadioField, SelectMultipleField, FieldList, widgets
from wtforms.validators import DataRequired, url

class ScatterGatherForm(FlaskForm):
    cluster_view = RadioField('cluster_view', choices=[(i,
                              'cluster_{}'.format(i)) for i in range(12)])
    cluster_select = SelectMultipleField('cluster_select', choices=[(i,
                              'cluster_{}'.format(i)) for i in range(12)],
                              option_widget=widgets.CheckboxInput(),
                              widget=widgets.ListWidget(prefix_label=False))
