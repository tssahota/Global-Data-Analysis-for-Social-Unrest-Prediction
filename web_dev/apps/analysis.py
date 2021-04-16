import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import glob
import os 

from app_temp import app
colorscale = [ "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]
# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)

path_list = ['./apps/analysis_data/'] # use your path
path = './apps/analysis_data/'
task_list = ['task1']
print("Print:", os.getcwd())

#Task1
df1 = pd.read_csv(path+'task1.csv')
#df2 = pd.read_csv(path+'master_gdelts_agg.csv')

years = df1['Year'].unique().tolist()[::-1]
year_list = [] 
for i in years:
    year_list.append({"label": i, "value": i})

countries = df1['ActionGeo_CountryCode'].unique().tolist()
countries = sorted(countries)
country_list = []
for i in countries:
    country_list.append({"label": i, "value": i})



# ------------------------------------------------------------------------------
## Define Figures
# Figure 1

# Initialize figure with subplots
df2 = df1.copy()
df3= df2.copy()
df3['MonthYear'] = df2['MonthYear'].astype(str) 
df3['Month']=df3['MonthYear'].apply(lambda x: x[4:])
df4 = df3.groupby('Month').mean().reset_index()
df5 = df3.groupby(['Month','EventRootCode']).count().reset_index()
df6= df3.groupby(['Month','EventRootCode']).mean().reset_index()
df5 = df5[['Month','EventRootCode','SQLDATE']]
df6 = df6[['Month','EventRootCode','GoldsteinScale']]
df5 = df5.merge(df6, on=['Month','EventRootCode'])

fig1_1 = px.scatter_geo(df2, lat=df2['ActionGeo_Lat'],lon=df2['ActionGeo_Long'],
                     hover_name="ActionGeo_CountryCode",  color='GoldsteinScale',
                     opacity=0.70,color_continuous_scale=px.colors.sequential.OrRd[::-1],
                     animation_frame="Year", #size = df2['GoldsteinScale'],
                     width=900, height=600,
                     template = 'plotly_dark',
                     projection="natural earth"
                     )
fig1_2 = make_subplots(
    rows=2, cols=1) 

fig1_2.add_trace(
    go.Scatter(x=df4['Month'],y=df4['AvgTone'], mode='lines', showlegend=False), #marker=dict(color="crimson"), 
    row=1, col=1)

fig1_2.add_trace(
    go.Scatter(x=df5['Month'],y=df5['GoldsteinScale'], mode='markers', 
        marker=dict(color=df5['EventRootCode']), showlegend=False), 
    row=2, col=1)

fig1_1.update_geos(
    #projection_type="orthographic",
    #landcolor="white",
    #oceancolor="MidnightBlue",
    #showocean=True,
    countrycolor='grey',
    #lakecolor="LightBlue"
    showcountries=True
)
fig1_2.update_layout(
    template = "plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    height = 600, width = 500
)

#Figure 2
# Initialize figure with subplots
fig2 = make_subplots(
    rows=2, cols=2,
    column_widths=[0.6, 0.4],
    row_heights=[0.4, 0.6],
    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar"}],
           [            None                    , {"type": "xy"}]])

# Add scattergeo globe map of volcano locations
fig2.add_trace(
    go.Scattergeo(lat=df2['ActionGeo_Lat'],lon=df2['ActionGeo_Long'],
                  mode="markers",
                  hoverinfo="text",
                  showlegend=False,
                  marker=dict(color="crimson", size=4, opacity=0.8)),
    row=1, col=1
)

# Add locations bar chart
fig2.add_trace(
    go.Bar(x=df5['EventRootCode'],y=df5['SQLDATE'], showlegend=False),
    row=1, col=2
)

# Add 3d surface of volcano
fig2.add_trace(
    go.Scatter(x=df2['Year'],y=df2['AvgTone'],mode='markers',
           showlegend=False),
    row=2, col=2
)
# Update geo subplot properties
fig2.update_geos(
    projection_type="orthographic",
    landcolor="white",
    oceancolor="MidnightBlue",
    showocean=True,
    lakecolor="LightBlue"
)

# Rotate x-axis labels
fig2.update_xaxes(tickangle=45)

# Set theme, margin, and annotation in layout
fig2.update_layout(
    template="plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    annotations=[
        dict(
            text="Source: NOAA",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=0)
    ]
)



"""
df = {}
for i, path in enumerate(path_list):
    filenames = glob.glob(path+"/*.parquet")
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_parquet(filename))
    # Concatenate all data into one DataFrame
    df[task_list[i]] = pd.concat(dfs, ignore_index=True)
"""
"""#732
genre_list = []
for genre_name in df['task3']['genre_name'].unique():
    genre_list.append({"label": genre_name, "value": genre_name})

year_list = []
year_list = []
for i in range(18):
    year_list.append({"label": 2000+(17-i), "value": 2000+(17-i)})
    year_list.append(2000 + i)

col_label = {}
col_label['vote_average'] = 'TMDB critics rating (Max=10)'
col_label['avg_user_rating'] = 'Average User Rating (Max=5)'
col_label['popularity'] = 'Popularity'
col_label['profit'] = 'Profit'

col_list = [{"label": col_label["popularity"], "value": "popularity"},
            {"label": col_label["profit"], "value": "profit"},
            {"label": col_label['vote_average'], "value": "vote_average"},
            {"label": col_label['avg_user_rating'], "value": "avg_user_rating"}]

"""
# task16_col_label = {}
# task16_col_label['vote_average'] = 'TMDB critics rating (Max=10)'
# task16_col_label['avg_user_rating'] = 'Average User Rating (Max=5)'
"""
job_list = []
for job_name in df['task16']['job'].unique():
    job_list.append({"label": job_name, "value": job_name})"""
# ------------------------------------------------------------------------------
# App layout
layout = html.Div([
    html.Div(id='task1_container', children=[
        html.H4(id='header_task1', children="Global Social Unrest Events Over The Years", style={'text-align': 'center'}),
        dcc.Graph(id='task1_map_chart', figure=fig1_1, style={'width': '65%', 'display': 'inline-block', 'float': 'left'}),
        dcc.Graph(id='task1_bar_chart', figure=fig1_2, style={'width': '35%','display': 'inline-block', 'float': 'right'}),
    ],style={'display': 'inline-block','padding-bottom': 50, 'vertical-align':'center'}),

    html.Div(id='task2_container', children=[
        html.H4(id='header_task2', children="Yearly Event Insights for a Country", style={'text-align': 'center'}),
        html.Div(id='task2_sub', children=[
            html.Div(id='task2_choice1', children=[  
                html.Label('Year:',style={'font-size':'12px'}),
                dcc.Dropdown(id="slct_year_task2",
                    options=year_list,
                    multi=False,
                    value='2021',
                    clearable=False,
                    style ={'background-color': '#101010', 'border-color':'#000000'}
                ),
            ], style={'display': 'inline-block', 'width':'50%', 'float':'left'},className="col-md-4"),
            html.Div(id='task2_choice2', children=[
                html.Label('Country:'),
                dcc.Dropdown(id="slct_country_task2",
                    options=country_list,
                    multi=False,
                    value='US',
                    clearable=False,
                    style ={'background-color': '#101010', 'border-color':'#000000'}#,'display': 'inline-block'}, 'vertical-align': 'left'}
                ),
            ], style={'display': 'inline-block', 'width':'50%', 'float':'right'}, className="col-md-5"),
        ], style={'width':'100%','padding-top': 25,'padding-bottom': 25},className="row"),
        dcc.Graph(id='task2_chart',figure=fig2, style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'center'#, 'margin': 0,'padding':0
            })
        ]),
    ])
"""
        html.Div(id='task4_container', children=[
            html.H2(id='header_task4', style={'text-align': 'center'}),
            html.Div(id='task4_sub', children=[
                html.Div(id='task4_choice', children=[
                html.Label('Parameter:'),
                dcc.Dropdown(id="slct_col_task4",
                    options=col_list,
                    multi=False,
                    value="popularity",
                    clearable=False,
                ),
                ], className="col-md-4"),
                html.Div(id='task4_p', children=[
                    html.P(
                        id="task4_insight",
                        children="List of Top 10 production companies with the highest selected parameter.",
                    ),
                ], className="col-md-8"),
            ], className="row"),
            dcc.Graph(id='task4_bar_chart')
        ]),
    
           html.Div(id='task16_container', children=[
            html.H2(id='header_task16', style={'text-align': 'center'}),
            html.Div(id='task2_sub', children=[
                html.Div(id='task2_choice', children=[
                    html.Label('Job:'),
                    dcc.Dropdown(id="slct_job_task16",
                        options=job_list,
                        multi=False,
                        value='Actor',
                        clearable=False
                    ),
                    # html.Label('Parameter:'),
                    # dcc.Dropdown(id="slct_col_task16",
                    #     options=col_list,
                    #     multi=False,
                    #     value="popularity",
                    #     clearable=False
                    # ),
                ], className="col-md-4"),
                html.Div(id='task16_p', children=[
                    html.P(
                        id="task16_insight",
                        children="List of Top 10 actors/directors with the highest average revenue generated.",
                    ),
                ], className="col-md-8"),
            ], className="row"),
            dcc.Graph(id='task16_bar_chart')
        ]),
"""

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
#***Task2 callback***
"""
@app.callback(
    [Output(component_id='header_task2', component_property='children'),
     Output(component_id='task2_chart', component_property='figure')],
    [Input(component_id='slct_year_task2', component_property='value'),
    Input(component_id='slct_country_task2', component_property='value')]   
)
"""
def update_graph(slct_year, slct_country):
    print('task1 update', slct_year, slct_col)
    container = f"Heading1_1"
    # filter column
    dff = df2[(df2['Year'] == int(slct_year)) & (df2['ActionGeo_CountryCode'] == slct_country)]
    
    # Add scattergeo globe map of volcano locations
    fig2.add_trace(
        go.Scattergeo(lat=dff['ActionGeo_Lat'],lon=dff['ActionGeo_Long'],
                      mode="markers",
                      hoverinfo="text",
                      showlegend=False,
                      marker=dict(color="crimson", size=4, opacity=0.8)),
        row=1, col=1
    )
    # Add locations bar chart
    fig2.add_trace(
        go.Bar(x=dff['ActionGeo_Lat'],y=dff['ActionGeo_Long'], #marker=dict(color="crimson"),
               showlegend=False),
        row=1, col=2
    )
    # Add 3d surface of volcano
    fig2.add_trace(
        go.Scatter(x=dff['ActionGeo_Lat'],y=dff['ActionGeo_Long'], #marker=dict(color="crimson"),
               showlegend=False),
        row=2, col=2
    )
    return container, fig2

"""
#***Task3 callback***
@app.callback(
    [Output(component_id='header_task2', component_property='children'),
    Output(component_id='task2_chart', component_property='figure')],
    [Input(component_id='slct_genre_task3', component_property='value'),
    Input(component_id='slct_col_task3', component_property='value')]
)
def update_graph(slct_genre, slct_col):
    print('task3 update', slct_genre, slct_col)
    container = f"Top 10 Highest {col_label[slct_col]} Movies in {slct_genre} (2000-2017)"
    dff = df["task3"].copy()
    #filter col
    dff = dff[dff["genre_name"] == slct_genre].sort_values(by=slct_col, ascending=False).head(10).sort_values(by=slct_col, ascending=True)
    #print("task3_dff", dff)
    fig = px.bar(data_frame=dff, y='title', x=slct_col, orientation='h', text=slct_col, template="plotly_white", color_continuous_scale=colorscale, color=slct_col)
    fig.update_layout(
        #title="Plot Title",
        xaxis_title=col_label[slct_col],
        yaxis_title='Title',
    )
    return container, fig

#***Task4 callback***
@app.callback(
    [Output(component_id='header_task4', component_property='children'),
    Output(component_id='task4_bar_chart', component_property='figure')],
    Input(component_id='slct_col_task4', component_property='value')
)
def update_graph(slct_col):
    print('task4 update')
    container = f"Top 10 Highest {col_label[slct_col]} Production Companies of All Time"
    dff = df["task4"].copy()
    #filter col
    dff = dff.sort_values(by=slct_col, ascending=False).head(10).sort_values(by=slct_col, ascending=True)
    #print("task4_dff", dff)
    fig = px.bar(data_frame=dff, y='production_company', x=slct_col, orientation='h', text=slct_col, template="plotly_white", color_continuous_scale=colorscale, color=slct_col)
    fig.update_layout(
        #title="Plot Title",
        xaxis_title=col_label[slct_col],
        yaxis_title='Production Company',
    )
    return container, fig

#***Task16 callback***
@app.callback(
    [Output(component_id='header_task16', component_property='children'),
    Output(component_id='task16_bar_chart', component_property='figure')],
    [Input(component_id='slct_job_task16', component_property='value')]
)
def update_graph(slct_job):
    print('task16 update', slct_job)
    container = f"Top 10 Highest Average Revenue Generating {slct_job}s of All Time"
    dff = df["task16"].copy()
    #filter col
    dff = dff[dff["job"] == slct_job].sort_values(by='avg_revenue', ascending=False).head(10).sort_values(by='avg_revenue', ascending=True)
    #print("task3_dff", dff)
    fig = px.bar(data_frame=dff, y='name', x='avg_revenue', orientation='h', text='avg_revenue', template="plotly_white", color_continuous_scale=colorscale, color='avg_revenue')
    fig.update_layout(
        xaxis_title='Average Revenue',
        yaxis_title='Name',
    )
    return container, fig
"""