import pandas as pd
import plotly.express as px
import dash
import json
import os
from dash import dcc, html, ctx, callback
from dash.dependencies import Input, Output

# Find all sessions
sessions = [folder for folder in os.listdir('sessions') if not os.path.isfile(folder) and os.path.isfile(f"sessions/{folder}/loss.json")]

# Select the latest session
select_session = sessions[-1]

# Load metadata & loss data from a specific session
def load_session_data(session: str):
    load_df = pd.read_csv(f'sessions/{session}/metadata.csv')
    with open(f'sessions/{session}/loss.json', 'r') as openfile:
        loss_data = json.load(openfile)
    print(len(loss_data['train_loss']), len(loss_data['test_loss']))
    load_loss_data_df = pd.DataFrame({ 'Train loss': loss_data['train_loss'] })
    load_test_data_df=pd.DataFrame({'Test loss': loss_data["test_loss"]})
    print(len(load_loss_data_df), len(load_test_data_df))
    return load_df, load_loss_data_df, load_test_data_df


df, loss_data_df,test_loss_data_df = load_session_data(select_session)

loss_fig = px.line(loss_data_df, title='Loss graph',labels={"index": "Epoch", "value": "Loss"})

# Remove unnecessary columns
df = df.drop(columns=['time', 'frame_id'])

df10=df.copy()

app = dash.Dash(__name__)

# Create an initial pie chart
fig = px.pie(df[df['country_code'] == 'SE'], names='road_condition', title='Metadata from cars', hole=0.3)


#De olika länderna
all_countries=[]
for i in df10["country_code"].unique():
    all_countries.append(i)
print(all_countries)

iso_alpha=["NOR","SWE","POL","DEU","LUX","FRA","IRL","GBR","ITA","FIN","HUN","CZE","NLD"]


#Antal länder
num_countries=[]
for i in all_countries:
  a=((df["country_code"]==i).sum())
  num_countries.append(a)




#Anomalier
anomaly_list=[]
for i in all_countries:
   a=df10[df["country_code"]==i]
   b=a[a["anomaly"]==-1]
   anomaly_list.append(len(b))




#Road conditions per land
different_road_conditions=df["road_condition"].unique()
road_conditions_country_normal=[]
road_conditions_country_wet=[]
road_conditions_country_snow=[]

for i in all_countries:
   a=df10[df10["country_code"]==i]
   for road_cond in different_road_conditions:
     b=a[a["road_condition"]==road_cond]
     if road_cond==different_road_conditions[0]:
        road_conditions_country_normal.append(len(b))
     elif road_cond==different_road_conditions[1]:
        road_conditions_country_wet.append(len(b))
     elif road_cond==different_road_conditions[2]:
        road_conditions_country_snow.append(len(b))

#Roadtype
different_road_types=df["road_type"].unique()
road_type_aerterial_rural=[]
road_type_higeway=[]
road_type_aerterial_urban=[]
road_type_city=[]
road_type_smaller_rural=[]
for i in all_countries:
   a=df10[df10["country_code"]==i]
   for road_type in different_road_types:
       b=a[a["road_type"]==road_type]
       if road_type==different_road_types[0]:
        road_type_aerterial_rural.append(len(b))
       elif road_type==different_road_types[1]:
        road_type_higeway.append(len(b))
       elif road_type==different_road_types[2]:
        road_type_aerterial_urban.append(len(b))
       elif road_type==different_road_types[3]:
        road_type_city.append(len(b))
       elif road_type==different_road_types[4]:
         road_type_smaller_rural.append(len(b))

data={"Country":all_countries,"Datapunkter":num_countries,"Iso_alpha":iso_alpha,"Anomalies":anomaly_list,"Road_condition_normal":road_conditions_country_normal,"Road_condition_wet":road_conditions_country_wet,"Road_condition_snow":road_conditions_country_snow,"Road_type_aerterial_rural":road_type_aerterial_rural,"road_type_higeway":road_type_higeway,"road_type_aerterial_urban":road_type_aerterial_urban,"road_type_city":road_type_city,"road_type_smaller_rural":road_type_smaller_rural}
df1=pd.DataFrame(data)

##----------------Antal Datapunkter per Land--------------------------------------------------------
fig4 = px.choropleth(df1, locations="Iso_alpha",
                    color="Datapunkter",
                    color_continuous_scale=px.colors.diverging.Portland,
                    title="Amount of datapoints, and Road Condition Data",
                    projection="natural earth",
                    scope="europe",
                    hover_data=["Road_condition_normal", "Road_condition_wet", "Road_condition_snow","Road_type_aerterial_rural","road_type_higeway","road_type_aerterial_urban","road_type_city","road_type_smaller_rural"])

fig4.update_traces(
    hovertemplate="<b>%{location}</b><br>"
                  "Datapoints: %{z}<br>"
                  "<b>Road Conditions</b><br>"
                  "Normal: %{customdata[0]}<br>"
                  "Wet: %{customdata[1]}<br>"
                  "Snow: %{customdata[2]}<extra></extra>"
                  "<b>Road Types</b><br>"
                  "Arterial Rural: %{customdata[3]}<br>"
                  "Highway: %{customdata[4]}<br>"
                  "Arterial Urban: %{customdata[5]}<br>"
                  "City: %{customdata[6]}<br>"
                  "Smaller Rural: %{customdata[7]}<extra></extra>"
                  ,text=df1['Country'])

fig4.add_annotation(
   text="Datapoints Per Country",
    x=.01,
    y=1,
    showarrow=False,
    font=dict(size=14, color="black")
)

for i in range(len(all_countries)):
   fig4.add_annotation(
    text=(iso_alpha[i]+": "+str(num_countries[i])),
    x=.01,
    y=(0.95-(i/100)*8),
    showarrow=False,
    font=dict(size=14, color="black")
   )

fig4.add_annotation(
   text="Total Amount of Datapoints: "+str(sum(num_countries)),
    x=.01,
    y=-.1,
    showarrow=False,
    font=dict(size=14, color="black")
)

##----------------Antal Anomalies per Land--------------------------------------------------------
fig5 = px.choropleth(df1, locations="Iso_alpha", color="Anomalies",
                    color_continuous_scale=px.colors.diverging.Picnic,
                    title="Amount of Anomalies per country",projection="natural earth",scope="europe")

fig5.update_traces(
    hovertemplate='<b>%{location}</b><br>Amount of Anomalies: %{z}',
    text=df1['Country']
)

for i in range(len(all_countries)):
   fig5.add_annotation(
    text=(iso_alpha[i]+": "+str(anomaly_list[i])),
    x=.01,
    y=(0.95-(i/100)*8),
    showarrow=False,
    font=dict(size=14, color="black")
   )

fig5.add_annotation(
   text="Total Amount of Anomalies: "+str(sum(anomaly_list)),
    x=.01,
    y=-.1,
    showarrow=False,
    font=dict(size=14, color="black")
)

geo_fig = px.scatter_geo(df,
                     lat=df['latitude'],
                     lon=df['longitude'],
                     color="anomaly", # which column to use to set the color of markers
                     hover_data=['road_condition', 'scraped_weather'],
                     projection="natural earth",
                     title="All the Anomalies",
                     scope="europe")
geo_fig.update_layout(
    height=800,
)

string1=("Total Amount of Anomalies: "+str(sum(anomaly_list)))
string2=("Total Amount of Datapoints: "+str(sum(num_countries)))
kalk=(sum(anomaly_list)/sum(num_countries))
kalk1=1-kalk
string3=("Anomalies: "+str(kalk)+"%")
string4=("Normal: "+str(kalk1)+"%")
lista1=[string1,string2,string3,string4]
counter=0
for i in lista1:
   geo_fig.add_annotation(
   text=i,
    x=.01,
    y=0.95-(counter/15),
    showarrow=False,
    font=dict(size=14, color="black"))
   counter +=1




# Create a scatter plot to show anomalies
# geo_fig = px.scatter_geo(df, lat=df['latitude'],
#                      lon=df['longitude'],
#                      color="anomaly", # which column to use to set the color of markers
#                      hover_data=['road_condition', 'scraped_weather'],
#                      projection="natural earth")
# geo_fig.update_layout(
#     height=800,
# )





##-------------------------------------------------------------------------------------------------------------------




# Create list of checklist items and remove unwanted columns
checklist_remove = ['num_vehicles', 'num_pedestrians', 'num_traffic_lights', 'anomaly_scores', 'anomaly']
checklist_items = [column for column in df.columns if column not in checklist_remove]

button_style = style={'width': '30%', 'margin-bottom': '5px', 'margin-top': '5px', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '12px'}

#Layout för pie charts och densitymap
app.layout = html.Div([
    # First row
    html.Div([
        dcc.Dropdown(sessions, sessions[-1], id='session', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': '32px'}),
        dcc.Graph(id='pie-chart', figure=fig),
        dash.html.H3(children='Categories', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto'}),
        dcc.Checklist(id='checklist', options=checklist_items, value=[checklist_items[0]], inline=True, style={'font-size': '16pt', 'width' : '70%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-bottom': '32px'}),
        # dcc.Dropdown(['Show all', 'Show inliers', 'Show outliers'], 'Show all', id='dropdown', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': '32px'}),
        # dcc.Graph(id='geo-map', figure=geo_fig),
        dash.html.H3(children='Map type', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto'}),
        html.Button('Datapoints', id='btn_nclicks_1', n_clicks=0, style=button_style),
        html.Button('Anomalies per country', id='btn_nclicks_2', n_clicks=0, style=button_style),
        html.Button('Anomalies coords', id='btn_nclicks_3', n_clicks=0, style=button_style),
        dcc.Graph(id='container-button-timestamp', figure=fig4),
        dcc.Graph(id='loss-graph', figure=loss_fig)
    ], style={'display': 'flex', 'flex-direction' : 'column'}),
])

@app.callback(
    [
        Output('container-button-timestamp', 'figure'),
        Output('pie-chart', 'figure'),
        # Output('geo-map', 'figure'),
        Output('loss-graph', 'figure')
    ],
    [
        Input('pie-chart', 'clickData'),
        Input('checklist', 'value'),
        # Input('dropdown', 'value'),
        Input('session', 'value'),
        Input('btn_nclicks_1', 'n_clicks'),
        Input('btn_nclicks_2', 'n_clicks'),
        Input('btn_nclicks_3', 'n_clicks'),
    ],
    prevent_initial_call=True
)


#Funktion för att uppdatera pie chartsen när man klickar på någon av de.
def update_pie_charts(clickData_pie_chart, checklist_values, session_value,btn_nclicks_1, btn_nclicks_2, btn_nclicks_3):
    global select_session
    global df
    global loss_data_df
    global test_loss_data_df
    figure = fig4
    if not session_value == select_session:
        select_session = session_value
        df, loss_data_df, test_loss_data_df = load_session_data(session_value)
    loss_fig = px.line(loss_data_df, title='Loss graph',labels={"index": "Epoch", "value": "Loss"})
    loss_fig.add_trace(px.line(test_loss_data_df,color_discrete_sequence=["orange"]).data[0])
    if len(checklist_values) == 0:
        updated_fig = px.pie(df, names=df.columns[0], title=f'Value distribution')
        return [figure, updated_fig, loss_fig]
    filtered_df = pd.DataFrame()
    filtered_df['selected_values'] = df[checklist_values].apply("-".join, axis=1)
    most_represented = filtered_df['selected_values'].mode()[0]
    pie_slices = len(filtered_df['selected_values'].unique())
    title = f'Selected fields: {" and ".join(checklist_values)} <br>Most represented: {most_represented.replace("-", " ")}'
    if pie_slices > 12:
        updated_fig = px.bar(
            filtered_df, 
            x='selected_values', 
            title=title
        )
    else:
        updated_fig = px.pie(
            filtered_df, 
            names='selected_values', 
            title=title
        )
    if "btn_nclicks_1" == ctx.triggered_id:
        figure=fig4
    elif "btn_nclicks_2" == ctx.triggered_id:
        figure=fig5
    elif "btn_nclicks_3" == ctx.triggered_id:
        figure=geo_fig  
    return [figure, updated_fig, loss_fig]

    if clickData_pie_chart is not None: #Om någon piechart blivit klickad på
        print(clickData_pie_chart)
        selected = clickData_pie_chart['points'][0]['label'] #Vilken del av pie charten som klickats på (ex. SE) 
        filtered_df = df[selected] #Nya dataframen baserat på vilket land man klickat på
        updated_fig = px.pie(filtered_df, names=df.columns, title=f'Distribution for {selected}')
        updated_figs['main'] = updated_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)  