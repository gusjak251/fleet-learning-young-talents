import pandas as pd
import plotly.express as px
import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output

df = pd.read_csv("metadata.csv")
with open('loss.json', 'r') as openfile:
    loss_data = json.load(openfile)
loss_data_df = pd.DataFrame({ 'loss': loss_data['train_loss']})

loss_fig = px.line(loss_data_df, title='Loss graph')

# Remove unnecessary columns
df = df.drop(columns=['time', 'frame_id'])

app = dash.Dash(__name__)

# Create an initial pie chart
fig = px.pie(df[df['country_code'] == 'SE'], names='road_condition', title='Metadata from cars', hole=0.3)

# Create a scatter plot to show anomalies
geo_fig = px.scatter_geo(df, lat=df['latitude'],
                     lon=df['longitude'],
                     color="anomaly", # which column to use to set the color of markers
                     hover_data=['road_condition', 'scraped_weather'],
                     projection="natural earth")
geo_fig.update_layout(
    height=800,
)


# Create list of checklist items and remove unwanted columns
checklist_remove = ['num_vehicles', 'num_pedestrians', 'num_traffic_lights', 'anomaly_scores', 'anomaly']
checklist_items = [column for column in df.columns if column not in checklist_remove]

#Layout för pie charts och densitymap
app.layout = html.Div([
    # First row
    html.Div([
        dcc.Graph(id='pie-chart', figure=fig),
        dash.html.H3(children='Categories', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto'}),
        dcc.Checklist(id='checklist', options=checklist_items, value=[checklist_items[0]], inline=True, style={'font-size': '16pt', 'width' : '70%', 'margin-left': 'auto', 'margin-right': 'auto'}),
        dcc.Dropdown(['Show all', 'Show inliers', 'Show outliers'], 'Show all', id='dropdown', style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': '32px'}),
        dcc.Graph(id='geo-map', figure=geo_fig),
        dcc.Graph(id='loss-graph', figure=loss_fig)
    ], style={'display': 'flex', 'flex-direction' : 'column'}),
])


#Callback för varje pie chart
@app.callback(
    [Output('pie-chart', 'figure'), Output('geo-map', 'figure')],
    [Input('pie-chart', 'clickData'), Input('checklist', 'value'), Input('dropdown', 'value')]
)

#Funktion för att uppdatera pie chartsen när man klickar på någon av de.
def update_pie_charts(clickData_pie_chart, checklist_values, dropdown_value):
    if dropdown_value == 'Show all':
        updated_geo_fig = px.scatter_geo(df, lat=df['latitude'],
                     lon=df['longitude'],
                     color="anomaly", # which column to use to set the color of markers
                     hover_data=['road_condition', 'scraped_weather'],
                     projection="natural earth")
    elif dropdown_value == 'Show inliers':
        inliers_df = df[df['anomaly'] == 1]
        updated_geo_fig = px.scatter_geo(inliers_df, lat=inliers_df['latitude'],
                     lon=inliers_df['longitude'],
                     color="anomaly", # which column to use to set the color of markers
                     hover_data=['road_condition', 'scraped_weather'],
                     projection="natural earth")
    elif dropdown_value == 'Show outliers':
        inliers_df = df[df['anomaly'] == -1]
        updated_geo_fig = px.scatter_geo(inliers_df, lat=inliers_df['latitude'],
                     lon=inliers_df['longitude'],
                     color="anomaly", # which column to use to set the color of markers
                     hover_data=['road_condition', 'scraped_weather'],
                     projection="natural earth")
    if len(checklist_values) == 0:
        updated_fig = px.pie(df, names=df.columns[0], title=f'Value distribution')
        return [updated_fig]
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
    return [updated_fig, updated_geo_fig]

    if clickData_pie_chart is not None: #Om någon piechart blivit klickad på
        print(clickData_pie_chart)
        selected = clickData_pie_chart['points'][0]['label'] #Vilken del av pie charten som klickats på (ex. SE) 
        filtered_df = df[selected] #Nya dataframen baserat på vilket land man klickat på
        updated_fig = px.pie(filtered_df, names=df.columns, title=f'Distribution for {selected}')
        updated_figs['main'] = updated_fig
    
    #Returnerar oavsett om någon klickats eller inte
    return updated_figs['main']

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)  