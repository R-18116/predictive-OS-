import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
import psutil
import platform
import socket
from sklearn.linear_model import LinearRegression

# Store history
history_data = {'time': [], 'cpu': [], 'memory': [], 'disk': [], 'battery': []}

# Predict future values
def predict_next(values, steps=10):
    if len(values) < 2:
        return [values[-1]] * steps
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    return model.predict(X_future)

# Line and Bar graph
def create_graph(title, real_values, pred_values):
    pred_time = pd.date_range(start=datetime.datetime.now(), periods=len(pred_values), freq='2s')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_data['time'], y=real_values, mode='lines+markers', name=f'{title}'))
    fig.add_trace(go.Scatter(x=pred_time, y=pred_values, mode='lines', name=f'{title} Prediction', line=dict(dash='dot')))
    fig.update_layout(title=f"{title} Usage Over Time", xaxis_title="Time", yaxis_title="%", template='plotly_dark', height=350)
    return fig

# System Info
def get_system_info():
    uptime_seconds = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime_str = str(uptime_seconds).split('.')[0]

    return {
        "OS Version": platform.platform(),
        "System Uptime": uptime_str,
        "Active Processes": len(psutil.pids()),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Total RAM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB",
        "Total Disk": f"{round(psutil.disk_usage('/').total / (1024 ** 3), 2)} GB",
        "System Name": platform.node(),
        "IP Address": socket.gethostbyname(socket.gethostname()),
    }

# App setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Layout
app.layout = html.Div(
    id='main-div',
    style={
        "backgroundImage": "url('https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1350&q=80')",
        "backgroundSize": "cover",
        "backgroundRepeat": "no-repeat",
        "backgroundAttachment": "fixed",
        "minHeight": "100vh",
        "padding": "10px"
    },
    children=[
        html.Div(style={
            "backgroundColor": "rgba(0, 0, 0, 0.75)",
            "padding": "20px",
            "borderRadius": "15px"
        }, children=[
            dbc.Container([
                html.H2("🖥️ Predictive OS Resource Allocation Dashboard", className="text-center text-light my-4"),

                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("CPU (%)"), dbc.CardBody(html.H4(id='cpu-text'))], color="primary", inverse=True), width=2),
                    dbc.Col(dbc.Card([dbc.CardHeader("Memory (%)"), dbc.CardBody(html.H4(id='memory-text'))], color="success", inverse=True), width=2),
                    dbc.Col(dbc.Card([dbc.CardHeader("Disk (%)"), dbc.CardBody(html.H4(id='disk-text'))], color="warning", inverse=True), width=2),
                    dbc.Col(dbc.Card([dbc.CardHeader("Uptime"), dbc.CardBody(html.H4(id='uptime-text'))], color="info", inverse=True), width=2),
                    dbc.Col(dbc.Card([dbc.CardHeader("Battery (%)"), dbc.CardBody(html.H4(id='battery-text'))], color="secondary", inverse=True), width=2),
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='cpu-graph'), width=6),
                    dbc.Col(dcc.Graph(id='memory-graph'), width=6),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='disk-graph'), width=6),
                    dbc.Col(dcc.Graph(id='battery-graph'), width=6),
                ]),

                html.H4("🔲 Process Table", className="text-light mt-4"),
                dbc.Button("Show/Hide Processes", id="toggle-process-btn", color="secondary", className="mb-2"),
                dcc.Store(id="process-visible", data=False),
                html.Div(id='process-table', className="text-light"),

                html.H4("💡 System Insights", className="text-light mt-4"),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody(html.H4("Tip: Keep your system updated!"))], color="primary", inverse=True), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody(html.H4("Warning: High memory usage detected!"))], color="warning", inverse=True), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody(html.H4("Tip: Close unused processes to free up resources."))], color="info", inverse=True), width=4),
                ]),

                dbc.Row([
                    dbc.Col(dbc.Switch(id='theme-toggle', label="Dark Mode", value=True), width=4),
                    dbc.Col(dcc.Dropdown(id='model-selector', options=[
                        {'label': 'Linear Regression', 'value': 'lr'},
                        {'label': 'Support Vector Machine', 'value': 'svm'}
                    ], value='lr', style={'color': 'black'}), width=4),
                    dbc.Col(dcc.Input(id='refresh-time', type='number', value=2, min=1, step=1, debounce=True), width=4),
                ], className="mb-4"),

                dbc.Button("🔄 Reset Graph", id="reset-graph-btn", color="danger", className="mb-4"),
                dcc.Store(id="reset-trigger", data=False),

                html.H6("📝 System Information & App Credits", className="text-light mt-4"),
                html.Div(id='system-info-table', className="text-light"),

                dcc.Interval(id='interval', interval=2000, n_intervals=0)
            ], fluid=True)
        ])
    ]
)

# Callbacks
@app.callback(
    Output('interval', 'n_intervals'),
    Input('refresh-time', 'value')
)
def update_refresh_rate(value):
    return 0

@app.callback(
    Output("process-visible", "data"),
    Input("toggle-process-btn", "n_clicks"),
    State("process-visible", "data"),
    prevent_initial_call=True
)
def toggle_process_table(n_clicks, visible):
    return not visible

@app.callback(
    Output("reset-trigger", "data"),
    Input("reset-graph-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_graph(n_clicks):
    history_data['time'].clear()
    history_data['cpu'].clear()
    history_data['memory'].clear()
    history_data['disk'].clear()
    history_data['battery'].clear()
    return True

@app.callback(
    [Output('cpu-graph', 'figure'), Output('memory-graph', 'figure'),
     Output('disk-graph', 'figure'), Output('battery-graph', 'figure'),
     Output('cpu-text', 'children'), Output('memory-text', 'children'),
     Output('disk-text', 'children'), Output('battery-text', 'children'),
     Output('uptime-text', 'children'),
     Output('process-table', 'children'), Output('system-info-table', 'children')],
    [Input('interval', 'n_intervals'), Input('process-visible', 'data'), Input('reset-trigger', 'data')]
)
def update_graph(n, show_processes, reset_trigger):
    now = datetime.datetime.now()
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    battery = psutil.sensors_battery().percent if psutil.sensors_battery() else 0
    system_info = get_system_info()

    history_data['time'].append(now)
    history_data['cpu'].append(cpu)
    history_data['memory'].append(memory)
    history_data['disk'].append(disk)
    history_data['battery'].append(battery)

    cpu_pred = predict_next(history_data['cpu'])
    mem_pred = predict_next(history_data['memory'])
    disk_pred = predict_next(history_data['disk'])
    batt_pred = predict_next(history_data['battery'])

    process_table = html.Div()
    if show_processes:
        processes = psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent'])
        process_rows = [html.Tr([html.Th("PID"), html.Th("Name"), html.Th("CPU (%)"), html.Th("Memory (%)")])]
        for proc in processes:
            process_rows.append(html.Tr([
                html.Td(proc.info['pid']),
                html.Td(proc.info['name']),
                html.Td(proc.info['cpu_percent']),
                html.Td(proc.info['memory_percent'])
            ]))
        process_table = html.Table(process_rows, className="table table-striped")

    return create_graph("CPU", history_data['cpu'], cpu_pred), \
           create_graph("Memory", history_data['memory'], mem_pred), \
           create_graph("Disk", history_data['disk'], disk_pred), \
           create_graph("Battery", history_data['battery'], batt_pred), \
           f"{cpu}%", f"{memory}%", f"{disk}%", f"{battery}%", \
           system_info.get("System Uptime"), \
           process_table, \
           html.Table([html.Tr([html.Td(k), html.Td(v)]) for k, v in system_info.items()])

if __name__ == '__main__':
    app.run(debug=True)
