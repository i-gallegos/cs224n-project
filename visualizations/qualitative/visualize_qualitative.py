import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

metrics = pd.read_csv('qualitative_metrics.csv')

ref_dim = go.parcats.Dimension(
    values=metrics['Reference Quality'],
    label="Reference Quality", categoryarray=[1, 2, 3],
    ticktext=['Poor', 'Moderate', 'Good']
)

qual_dim = go.parcats.Dimension(
    values=metrics['Prediction Quality'],
    label="Prediction Quality", categoryarray=[1, 2, 3],
    ticktext=['Poor', 'Moderate', 'Good']
)

match_dim = go.parcats.Dimension(
    values=metrics['Prediction Match'],
    label="Prediction Match", categoryarray=[1, 2, 3],
    ticktext=['Poor', 'Moderate', 'Good']
)

# Create parcats trace
color = metrics['Prediction Match']

fig = go.Figure(data = [go.Parcats(dimensions=[ref_dim, qual_dim, match_dim],
        line={'color': color},
        arrangement='freeform')])

fig.write_image(file='qualitative.png', format='png')
