import dash
from dash import dcc, html
import dash_table
import pandas as pd
from config import config

# Sample DataFrame with Premier League team odds
df = pd.read_csv(config.sim_output_file)


# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    [
        html.H1("Premier League Odds Tracker"),
        html.Div(
            "This table shows the latest odds for winning the Premier League, finishing in the top 4, and relegation."
        ),
        dash_table.DataTable(
            id="odds-table",
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict("records"),
            style_table={"height": "400px", "overflowY": "auto"},
            style_cell={
                "textAlign": "center",
                "font-family": "Arial, sans-serif",
                "padding": "8px",
                "border": "1px solid #ddd",
            },
            style_header={
                "backgroundColor": "#f4f4f9",
                "fontWeight": "bold",
                "fontSize": "14px",
            },
            style_data_conditional=[
                {
                    "if": {
                        "column_id": "title_odds",
                        "filter_query": "{Title Odds} > 0.5",
                    },
                    "backgroundColor": "green",
                    "color": "white",
                },
                {
                    "if": {
                        "column_id": "title_odds",
                        "filter_query": "{Title Odds} <= 0.5",
                    },
                    "backgroundColor": "red",
                    "color": "white",
                },
                {
                    "if": {
                        "column_id": "top_4_odds",
                        "filter_query": "{Top 4 Odds} > 0.8",
                    },
                    "backgroundColor": "blue",
                    "color": "white",
                },
                {
                    "if": {
                        "column_id": "relegation_odds",
                        "filter_query": "{Relegation Odds} > 0.05",
                    },
                    "backgroundColor": "orange",
                    "color": "black",
                },
            ],
            sort_action="native",  # Enables sorting
            sort_mode="multi",  # Allow multiple sorting
            page_action="native",  # Paginate the data
            page_size=20,  # Set the number of rows per page
        ),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
