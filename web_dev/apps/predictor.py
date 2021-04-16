import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app_temp import app
import pandas as pd
import pandas as pd
import plotly.express as px
from datetime import date
#import plotly.graph_objects as go


# Import Data
path = './apps/analysis_data/'
df = pd.read_csv(path+'live.csv')

# Define Figures
"""
fig1 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [None, {"rowspan": 2}]],
"""
df1 = df.copy()


df1=df1[df1['SQLDATE']>20210301]
std_avg = df1['GoldsteinScale'].mean()
std_max = df1['GoldsteinScale'].max()
df1['GoldsteinScale_std'] = df1['GoldsteinScale'] - std_avg

fig1 = px.choropleth(df1, locations="Actor1CountryCode",
                    color='GoldsteinScale_std', # lifeExp is a column of gapminder
                    hover_name="Actor1CountryCode", # column to add to hover information
                    locationmode="ISO-3",
                    hover_data = ['GoldsteinScale','AvgTone'],
                    color_continuous_scale=px.colors.diverging.RdBu)

fig1.update_layout(
    template = "plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    #height = 600, width = 425
)
#features = ['feature1', 'vote_count','popularity', 'keyword_power', 'youtube_views', 'youtube_likes']
# App layout
layout = html.Div([
    html.Div(id='predictor_container', children=[
        html.H2(id='header_predictor', children='Global Social Unrest Predictor', style={'text-align': 'center'}),
        html.Div(id='predictor_sub', children=[
            html.Div(id='predictor_ui', children=[
                html.Div(children=[
                html.Div(children=[
                    dcc.DatePickerRange(
                    id='input-date',
                    min_date_allowed=date(2021, 4,12 ),
                    max_date_allowed=date(2021, 5, 12),
                    initial_visible_month=date(2021, 4, 12),
                    end_date=date(2021, 5, 12)
                    ),
                    #html.Div(id='output-date'),
                ]),
                html.Div(children=[
                    dbc.Button("Predict", id="predict_btn", color="primary", className="ml-5 float-right")
                ]),
            ], style={'margin-top': '15px'}),
        ], className="col-md-8"),
        dcc.Graph(id='chloro-map', figure=fig1, style={"margin": "0", "width" : '100%'})           
        ], className="row"),
]),
    ])


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
