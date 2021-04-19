import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app_temp import app
import pandas as pd
import plotly.express as px
from datetime import date
import pickle
import pomegranate
import plotly.graph_objects as go
import statsmodels.api as sm
from PIL import Image
import plotly.graph_objects as go
import numpy 
import mchmm as mc 
import os
os.environ["PATH"] += os.pathsep + '/Users/apple/anaconda3/lib/python3.6/site-packages/graphviz/'

# Import Data
last_n = 0 # Global click count variable init 
last_n2 = 0 # Global click count variable init 


path = './apps/analysis_data/'
model_path = './apps/models/'
df = pd.read_csv(path+'live.csv')

df1=df.dropna()
df1=df1[['Actor1CountryCode','GoldsteinScale','AvgTone','SQLDATE']]
df2=df1.groupby('Actor1CountryCode').mean().reset_index()
std_avg = df2['GoldsteinScale'].mean()
df2['Goldstein Scale'] = df2['GoldsteinScale'] - std_avg

fig1 = px.choropleth(df2, locations="Actor1CountryCode",
                    color='Goldstein Scale', # lifeExp is a column of gapminder
                    hover_name="Actor1CountryCode", # column to add to hover information
                    locationmode="ISO-3",
                    hover_data = ['Goldstein Scale','AvgTone'],
                    color_continuous_scale=px.colors.diverging.RdBu)

fig1.update_layout(
    template = "plotly_dark",
    margin=dict(r=5, t=5, b=5, l=5),
    #height = 600, width = 425
)

# Create figure
img_width = 1400
img_height = 1400
scale_factor = 0.9

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=[0, img_width * scale_factor],
        y=[0, img_height * scale_factor],
        mode="markers",
        marker_opacity=0))

fig3.update_xaxes(
    visible=False,
    range=[0, img_width * scale_factor]
)

fig3.update_yaxes(
    visible=False,
    range=[0, img_height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
)
# Set templates

fig3.update_layout(template="plotly_dark")

# Load events model
filename = model_path+'event_model.sav'
event_model = pickle.load(open(filename, 'rb'))


filename2 = model_path+'gs_model.sav'
gs_model = pickle.load(open(filename2, 'rb'))


#-----------------------------------------------------------------------------------------------------------------
# App layout
layout = html.Div([
    html.Div(id='gs_container', children=[
        html.Div(id='gs_container', children=[
        html.H4(id='header_gs', children="Global Social Unrest Prediction", style={'text-align': 'center'}),
        html.Div(id='gs_sub', children=[ 
            html.Div(children=[
                html.Label('Start Date',style ={'font-size':19,'fontSize':19}),
                dcc.Input(
                    id="from_date",
                    placeholder="DD/MM/YYYY",
                    type='text',
                    style ={'background-color': '#101010', 'border-color':'#000000','fontSize':18, 'fontColor':' #e6e6e6', 'color':' #e6e6e6'}
                ),]
                ,style={'display': 'inline-block', 'width':'20%', 'float':'left'}, className="col-md-5"),
            html.Div(children=[
                html.Label('End Date       ',style ={'font-size':19,'fontSize':19}),
                dcc.Input(
                    id="to_date",
                    placeholder="DD/MM/YYYY",
                    type='text',
                    style ={'background-color': '#101010', 'border-color':'#000000','fontSize':18, 'fontColor':' #e6e6e6', 'color':' #e6e6e6', 'width': '80%'}
                ),]
                ,style={'display': 'inline-block', 'width':'20%', 'float':'left'}, className="col-md-5"),
            html.Div(id='button_1', children=[
                html.Label('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', style={'color':'#000000'}),
                dbc.Button("Predict", id="predict_btn", color="primary", className="col-md-5", style={'width':'80%', 'vertical-align':'right'})
            ],style={'display':'block','width':'20%', 'float':'right'}, className="col-md-5"),
        ], style={'display': 'inline-block','width':'100%','padding-top': 25,'padding-bottom': 25},className="row"),
        dcc.Graph(id='gs_chart',figure=fig1, style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'center'#, 'margin': 0,'padding':0
            })
        ]),
    html.Div(id='events_container', children=[
        html.Div(id='events_container', children=[
        html.H4(id='header_events', children="Event Sequence Prediction", style={'text-align': 'center'}),
        html.Div(id='events_sub', children=[ 
            html.Div(children=[
                html.Label('Enter Sequence',style ={'font-size':19,'fontSize':19}),
                dcc.Input(
                    id="sequence",
                    placeholder="ABBCDDE",
                    type='text',
                    style ={'background-color': '#101010', 'border-color':'#000000', 'fontSize':18, 'fontColor':' #e6e6e6', 'color':' #e6e6e6'}
                ),]
                ,style={'width':'20%', 'float':'left'}, className="col-md-5"),
            html.Div(children=[
                html.Label('Prediction',style ={'font-size':19,'fontSize':19}),
                dcc.Input(
                    id="event_pred",
                    placeholder=" ",
                    type='number',
                    readOnly=True,
                    style ={'background-color': '#101010', 'border-color':'#000000','fontSize':18, 'fontColor':' #e6e6e6', 'color':' #e6e6e6'}
                ),]
                ,style={ 'width':'20%', 'float':'left'}, className="col-md-5"),
            html.Div(id='button_1', children=[
                html.Label('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', style={'color':'#000000'}),
                dbc.Button("Predict", id="predict_btn2", color="primary", className="col-md-5", style={'width':'80%', 'vertical-align':'right'})
            ],style={'display':'block','width':'20%', 'float':'right'}, className="col-md-5"),
        ], style={'display': 'inline-block','width':'100%','padding-top': 25,'padding-bottom': 25},className="row"),
        dcc.Graph(id='event_chart',figure=fig3, style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'center'#, 'margin': 0,'padding':0
            })
        ], style={'padding-top': 25},),
    ]),
    ]),
])    

# ------------------------------------------------------------------------------
@app.callback(
    Output(component_id="gs_chart", component_property="figure"),
    [Input(component_id="from_date", component_property="value"),
    Input(component_id="to_date", component_property="value"),
    Input("predict_btn", "n_clicks")
    ],
)
def update_graph(from_date, to_date, n):
    global last_n
    global fig1
    print('n here', n, last_n)
    if (n is not None):
        if (n>last_n):
            last_n = last_n + 1
            dff = df1.copy()
            # convert input dates to comparable format
            from_sqldate = int(from_date[6:] + from_date[3:5] + from_date[0:2])
            to_sqldate = int(to_date[6:] + to_date[3:5] + to_date[0:2])
            print('gs update', from_date, to_date, from_sqldate, to_sqldate)

            #dff_1= gs_model.predict(dff[[features]])
            # Apply filters
            dff=dff[(dff['SQLDATE']>from_sqldate) & (dff['SQLDATE']<to_sqldate)]
            print(dff.count())
            dff=dff.groupby('Actor1CountryCode').mean().reset_index()
            std_avg = dff['GoldsteinScale'].mean()

            # Standardize Goldstein value
            dff['Goldstein Scale'] = dff['GoldsteinScale'] - std_avg

            fig1 = px.choropleth(dff, locations="Actor1CountryCode",
                                color='Goldstein Scale', # lifeExp is a column of gapminder
                                hover_name="Actor1CountryCode", # column to add to hover information
                                locationmode="ISO-3",
                                hover_data = ['Goldstein Scale','AvgTone'],
                                color_continuous_scale=px.colors.diverging.RdBu)

            fig1.update_layout(
                template = "plotly_dark",
                margin=dict(r=10, t=25, b=40, l=60),
            )
    return fig1

@app.callback(
    [Output(component_id="event_chart", component_property="figure"),
     Output(component_id="event_pred", component_property="value")],
    [Input(component_id="sequence", component_property="value"),
    Input("predict_btn2", "n_clicks")
    ],
)
def update_graph(inp_seq, n):
    global last_n2
    global fig3
    out_seq=""
    print('n here', n, last_n2)
    if (n is not None):
        if (n>last_n2):
            last_n2 = last_n2 + 1
            print('In seq', inp_seq)
            seq = numpy.array(list(inp_seq)) 
            hmm_pred = event_model.predict(seq)
            out_seq = ''.join(map( str, hmm_pred))
            print(out_seq)

            sim = mc.MarkovChain().from_data(inp_seq) 
            graph = sim.graph_make(
                      format="png",
                      graph_attr=[("rankdir", "LR"),("bgcolor","transparent")],
                      node_attr=[("fontname", "Sans bold"), ("fontsize", "20"), ("fontcolor","white"), ("color","white")],
                      edge_attr=[("fontname", "Sans bold"), ("fontsize", "12"), ("fontcolor","grey"), ("color","grey")]
                    )

            graph.render('./img/m_chain')
            img3= Image.open('./img/m_chain.png')

            # Clear layout and add new graphs
            #fig3.layout = {}
            fig3.add_layout_image(
                dict(
                    x=0,
                    sizex=img_width * scale_factor,
                    y=img_height * scale_factor,
                    sizey=img_height * scale_factor,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=img3))

            fig3.update_layout(
                template = "plotly_dark")

    return fig3, out_seq

