import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app_temp import app
import pandas as pd
import plotly.express as px
from datetime import date
import json
import pickle
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from os import listdir
from os.path import isfile, join
#import plotly.graph_objects as go


def create_graph(graph, title, pos_):
# Custom function to create an edge between node x and node y, with a given text and width
    def make_edge(x, y, text, width):
        return  go.Scatter(x         = x,
                           y         = y,
                           line      = dict(width = width,
                                       color = '#fca311'),
#                            hoverinfo = 'text',
#                            text      = ([text]),
                           mode      = 'lines')

    edge_trace = []
    for edge in graph.edges():
        char_1 = edge[0]
        char_2 = edge[1]
        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]
        text   = char_1 + '--' + char_2 + ': ' + str(graph.edges()[edge]['weight'])

        trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                           width = 0.3*graph.edges()[edge]['weight']**1.75)
        edge_trace.append(trace)

    # Make a node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "top center",
                            textfont_size = 8,
                            mode      = 'markers+text',
                            hoverinfo = 'none',
                            marker    = dict(color = [],
                                             size  = [],
                                             line  = None))
    # For each node in midsummer, get the position and size and add to the node_trace
    for node in graph.nodes():
        x, y = pos_[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple(['#fca311'])
        node_trace['marker']['size'] += tuple([5*graph.nodes()[node]['size']])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])

    # Customize layout
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)', # transparent background
        plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
        yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
    )
    # Create figure
    fig = go.Figure(layout = layout )
    fig.update_layout(title_text=title, title_x=0.5)
    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    # Add node trace mulitple times for
    fig.add_trace(node_trace)
    fig.add_trace(node_trace)
    fig.add_trace(node_trace)
    # Remove legend
    fig.update_layout(showlegend = False)
    # Remove tick labels
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    # Show figure
    return fig

# Import Data
path = './apps/twitter_data/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
country_set = set()
file_list_dict = {}
for filename in onlyfiles:
    if (filename.endswith('.pkl')) and ('related_hashtags' in filename or 'trending_hashtags' in filename):
        splits = filename.split('_')
        if len(splits) > 2:
            country = splits[2]
            country_set.add(country)
            if country in file_list_dict.keys():
                temp_dict = file_list_dict[country]
            else:
                temp_dict = {}
            if 'trending_hashtags' in filename:
                temp_dict['trending_hashtags'] = filename
            else:
                temp_dict['related_hashtags'] = filename
            file_list_dict[country] = temp_dict

countries = list(country_set)
country_list = []
for country in countries:
    country_list.append({"label": country, "value": country})
# country = countries[0]
trending_hashtag_file = open(path+file_list_dict[country]['trending_hashtags'], "rb")
trending_hashtags = pickle.load(trending_hashtag_file)
related_hashtags_file = open(path+file_list_dict[country]['related_hashtags'], "rb")
related_hashtags_dict = pickle.load(related_hashtags_file)

hashtag = list(related_hashtags_dict.keys())[0]
related_hashtags = related_hashtags_dict[hashtag]

hashtag_list = []
for tag in related_hashtags_dict.keys():
    hashtag_list.append({"label": tag, "value": tag})

# Define Figures
G = nx.Graph()
G.add_node(country, size=10)
count=1
max_size = 5
factor = max_size/len(trending_hashtags)
for trend in trending_hashtags:
    G.add_node(trend[0], size=count*factor+1)
    G.add_edge(country, trend[0], weight = count*factor)
    count = count + 1

nodes = list(G.nodes(data=True))
edges = list(G.edges(data=True))
np.random.shuffle(nodes)

graph=nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
pos_ = nx.circular_layout(graph, scale = 1)
pos_[country] =  np.array([0, 0])
title = 'TRENDING HASHTAGS'
fig1 = create_graph(graph, title, pos_)
fig1.update_layout(
    template = "plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    #height = 600, width = 425
)

G = nx.Graph()
G.add_node(hashtag, size=10)
count=1
max_size = 5
factor = max_size/len(related_hashtags)
for tag in related_hashtags:
    G.add_node(tag[0], size=count*factor+1)
    G.add_edge(hashtag, tag[0], weight = count*factor)
    count = count + 1

nodes = list(G.nodes(data=True))
edges = list(G.edges(data=True))
np.random.shuffle(nodes)

graph=nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
pos_ = nx.circular_layout(graph, scale = 1)
pos_[hashtag] =  np.array([0, 0])
title = 'RELATED HASHTAGS'
fig2 = create_graph(graph, title, pos_)
fig2.update_layout(
    template = "plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    #height = 600, width = 425
)
# df1 = df.copy()
#
#
# df1=df1[df1['SQLDATE']>20210301]
# std_avg = df1['GoldsteinScale'].mean()
# std_max = df1['GoldsteinScale'].max()
# df1['GoldsteinScale_std'] = df1['GoldsteinScale'] - std_avg
#
# fig1 = px.choropleth(df1, locations="Actor1CountryCode",
#                     color='GoldsteinScale_std', # lifeExp is a column of gapminder
#                     hover_name="Actor1CountryCode", # column to add to hover information
#                     locationmode="ISO-3",
#                     hover_data = ['GoldsteinScale','AvgTone'],
#                     color_continuous_scale=px.colors.diverging.RdBu)


#features = ['feature1', 'vote_count','popularity', 'keyword_power', 'youtube_views', 'youtube_likes']
# App layout
layout = html.Div([
    html.Div(id='predictor_container', children=[
        # html.H4(id='header_predictor', children='Global Twitter Data Analysis', style={'text-align': 'center'}),
        html.Div(id='predictor_sub', children=[
            html.H5(id='header_task1', children="Top Trending Hashtags Countrywise", style={'text-align': 'center'}, className="col-md-12"),
            html.Div(id='predictor_ui', children=[
                html.Div(children=[
                html.Div(id='task1_choice1', children=[
                    html.Label('Country:',style={'font-size':'12px'}),
                    dcc.Dropdown(id="slct_country_task1",
                        options=country_list,
                        multi=False,
                        value=country,
                        clearable=False,
                        style ={'background-color': '#101010', 'border-color':'#000000'}#,'display': 'inline-block'}, 'vertical-align': 'left'}
                    ),
                ], style={'display': 'inline-block', 'width':'100%', 'float':'left'}, className="col-md-4"),
            ], style={'width': '100%', 'padding-top': 25,'padding-bottom': 25, 'float':'left'}),
        ], className="col-md-12", style={"margin": "0", "width" : '100%'}),
        dcc.Graph(id='scatter-map-1', figure=fig1, style={"margin": "0", "width" : '100%'}),
        html.Div(id='related_hashtags_ui', children=[
            html.H5(id='header_task1', children="Related Hashtags for automatically detected unrest related hashtags", style={'text-align': 'center'}),
            html.Div(children=[
            html.Div(id='task2_choice1', children=[
                html.Label('Hashtag:',style={'font-size':'12px'}),
                dcc.Dropdown(id="slct_hashtags_task2",
                    options=hashtag_list,
                    multi=False,
                    value=hashtag,
                    clearable=False,
                    style ={'background-color': '#101010', 'border-color':'#000000'}#,'display': 'inline-block'}, 'vertical-align': 'left'}
                ),
            ], style={'display': 'inline-block', 'width':'100%', 'float':'left'}, className="col-md-4"),
        ], style={'width': '100%', 'padding-top': 25,'padding-bottom': 25}),
     ], className="col-md-12", style={"margin": "0", "width" : '100%'}),
        dcc.Graph(id='scatter-map-2', figure=fig2, style={"margin": "0", "width" : '100%'})
        ], className="row"),
]),
    ])

@app.callback(
    [Output(component_id="scatter-map-1", component_property="figure"),
    Output(component_id="slct_hashtags_task2", component_property="options"),
    Output(component_id="slct_hashtags_task2", component_property="value"),],
    Input(component_id="slct_country_task1", component_property="value"),
)
def update_figure_1(value):
    trending_hashtag_file = open(path+file_list_dict[value]['trending_hashtags'], "rb")
    trending_hashtags = pickle.load(trending_hashtag_file)
    related_hashtags_file = open(path+file_list_dict[value]['related_hashtags'], "rb")
    related_hashtags_dict = pickle.load(related_hashtags_file)
    hashtag_list = []
    for tag in related_hashtags_dict.keys():
        hashtag_list.append({"label": tag, "value": tag})

    G = nx.Graph()
    G.add_node(value, size=10)
    count=1
    max_size = 5
    factor = max_size/len(trending_hashtags)
    for trend in trending_hashtags:
        G.add_node(trend[0], size=count*factor+1)
        G.add_edge(value, trend[0], weight = count*factor)
        count = count + 1

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    np.random.shuffle(nodes)

    graph=nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos_ = nx.circular_layout(graph, scale = 1)
    pos_[value] =  np.array([0, 0])
    title = 'TRENDING HASHTAGS'
    fig1 = create_graph(graph, title, pos_)
    fig1.update_layout(
        template = "plotly_dark",
        margin=dict(r=10, t=25, b=40, l=60),
        #height = 600, width = 425
    )
    return fig1,hashtag_list, list(related_hashtags_dict.keys())[0]

@app.callback(
    Output(component_id="scatter-map-2", component_property="figure"),
    [Input(component_id="slct_hashtags_task2", component_property="value"),
    Input(component_id="slct_country_task1", component_property="value")],
)
def update_figure_2(hashtag,value):
    related_hashtags_file = open(path+file_list_dict[value]['related_hashtags'], "rb")
    related_hashtags_dict = pickle.load(related_hashtags_file)
    related_hashtags = related_hashtags_dict[hashtag]
    related_hashtags = sorted(related_hashtags.items(), key = lambda kv:(kv[1], kv[0]))
    G = nx.Graph()
    G.add_node(hashtag, size=10)
    count=1
    max_size = 5
    factor = max_size/len(related_hashtags)
    for tag in related_hashtags:
        G.add_node(tag[0], size=count*factor+1)
        G.add_edge(hashtag, tag[0], weight = count*factor)
        count = count + 1

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    np.random.shuffle(nodes)

    graph=nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos_ = nx.circular_layout(graph, scale = 1)
    pos_[hashtag] =  np.array([0, 0])
    title = 'RELATED HASHTAGS'
    fig2 = create_graph(graph, title, pos_)
    fig2.update_layout(
        template = "plotly_dark",
        margin=dict(r=10, t=25, b=40, l=60),
        #height = 600, width = 425
    )
    return fig2

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
# @app.callback(
#     Output(component_id="director", component_property="options"),
#     [Input(component_id="director", component_property="value"),],
# )
# def update_dropdown_options(values):
#     if values and len(values) == 1:
#         return [option for option in director_options if option["value"] in values]
#     else:
#         return director_options

# @app.callback(
#     Output(component_id="genre", component_property="options"),
#     [Input(component_id="genre", component_property="value"),],
# )
# def update_dropdown_options(values):
#     if values and len(values) == 3:
#         return [option for option in genre_options if option["value"] in values]
#     else:
#         return genre_options

# @app.callback(
#     Output(component_id="cast", component_property="options"),
#     [Input(component_id="cast", component_property="value"),],
# )
# def update_dropdown_options(values):
#     if values and len(values) == 3:
#         return [option for option in cast_options if option["value"] in values]
#     else:
#         return cast_options
"""
features = ['feature1', 'vote_count','popularity', 'keyword_power', 'youtube_views', 'youtube_likes']

@app.callback(
    Output(component_id="chloro-map", component_property="value"),
    [Input(component_id="feature1", component_property="value"),
    Input(component_id="input-date", component_property="value"),
    Input("predict_btn", "n_clicks")
    ],
)
def update_graph(slct_genre, slct_col):
    print('task3 update', slct_genre, slct_col)
    #container = f"Top 10 Highest {col_label[slct_col]} Movies in {slct_genre} (2000-2017)"
    dff = df1.copy()
    #filter col
    #dff = dff[dff["genre_name"] == slct_genre].sort_values(by=slct_col, ascending=False).head(10).sort_values(by=slct_col, ascending=True)
    dff = dff.head(5000)
    #print("task3_dff", dff)
        #title="Plot Title",
    fig2 = px.choropleth(df1, locations="Actor1CountryCode",
                    color='GoldsteinScale_std', # lifeExp is a column of gapminder
                    hover_name="Actor1CountryCode", # column to add to hover information
                    locationmode="ISO-3",
                    hover_data = ['GoldsteinScale','AvgTone'],
                    color_continuous_scale=px.colors.diverging.RdBu
                    )
    return container, fig2

def predict_features(feature1, vote_count,popularity, keyword_power, youtube_views, youtube_likes, n):
    # if release_date is not None:
    #     date_object = date.fromisoformat(release_date)
    #     date_string = date_object.strftime('%B %d, %Y')
    #     print (string_prefix + date_string)
    global last_pred
    global last_n
    print('n here', n, last_n)
    if n:
        if n - 1 == last_n:
            last_n = n
            print('udpate n here', n, last_n)
            features_res = {'revenue':7, 'collection':8}
            temp = [feature1, vote_count, popularity, keyword_power, youtube_views, youtube_likes]
            label = ['feature1', 'Vote Count', 'Popularity', 'Keyword Power', 'Youtube Views', 'Youtube Likes']
            mis_list = []
            for i, feature in enumerate(features):
                # if i == len(temp)-1:
                #     #print(temp[i])
                #     features_res[feature] = datetime.strptime(temp[i], '%Y-%m-%d').timetuple().tm_yday
                # else:
                if temp[i] == None:
                    mis_list.append(label[i])
                features_res[feature] = temp[i]
            #update predict result
            if mis_list:
                err_msg = 'Error: Please fill in '
                for ele in mis_list:
                    err_msg = err_msg + ele + ', '
                return err_msg+' and try again.'
            else:
                #temp_res = {'feature1': 1, 'vote_count':2, 'popularity':3, 'keyword_power':4, 'youtube_views':5, 'youtube_likes':6, 'revenue':7, 'collection':8}
                sc_df = spark.createDataFrame(Row(**i) for i in [features_res])
                sc_df.show()
                predictions = model.transform(sc_df)
                predictions.show()
                prediction = predictions.collect()[0].asDict()['prediction']
                last_pred = prediction
                return prediction

        else:
            return last_pred
    else:
        return None
"""
