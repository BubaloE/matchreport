import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch, Bumpy, FontManager
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter1d
from mplsoccer import VerticalPitch
import cmasher as cmr
import matplotlib.cm as cm
import matplotlib.image as mpimg

# Konfiguration af siden
st.set_page_config(page_title="Match Report", page_icon="📊")

@st.cache_data
def fetch_team_data():
    teams_url = "http://api.performfeeds.com/soccerdata/team/1vzv4a2oaoik71bcxuaul4qsjs?_rt=b&tmcl=1zqurbs9rmwtk30us5y1v1rtg"
    teams_response = requests.get(teams_url)
    root = ET.fromstring(teams_response.content)

    contestants = []
    for contestant in root.findall(".//contestant"):
        contestant_data = {
            "id": contestant.get("id"),
            "name": contestant.get("name"),
            "officialName": contestant.get("officialName")
        }
        contestants.append(contestant_data)

    contestants_df = pd.DataFrame(contestants)

    # Definér mapping fra id til TeamId
    id_to_teamid = {
        "aljua8crr0vkwit0j3m59jw9j": "1rr41tgqg6pdzps45l79q2xn9", # Sønderjyske
        "b3r6d8ydtmtjckam6pku9y400": "cdyq2cl21zmti7yxyednecp91",
        "anga7587r1zv4ey71ge9zxol3": "1hl7veuy8npulidzmc6pz4wk5",
        "9qsmopgutr7ut5g6workk8w4i": "ai2srubk366phz9h7lw615uxh", # Brøndby
        "c165yjiny1qnmfdvefxvflnkc": "b6fr4n8833taghtqlm76dcjx1", # Vejle
        "77tfx9me4aaqhzv78bmgsy9bg": "48xw7kyik3l0wi1qzny2i8q95",
        "9w3sjm5ml4sjr2v81oxzdrt8v": "9ktd7jnil5sh965h7j0fuod4a", # Randers
        "ahnt9ejgwj58ul8auzdvi3czf": "1fftxw79d5tjl0rzdffhtcr11",
        "cftj6g399tknheudaxsdpfdgb": "2chzathpj8o0eoxedlxkklc5x",
        "36g6ifzjliec1jqnbtf7yesme": "3xvocl5rk3rkgilofj2fplkyy",
        "59as3grjvj19voay31j3yfgni": "8j0z9a7uit79ntr8d1xcc8wyd",
        "5rz9enoyknpg8ji78za5b82p0": "eenms6qw5avbs70mjrmbvldxx"
    }

    # Definér mapping fra name til ImageURL
    name_to_imageurl = {
        "Lyngby": "https://images.seeklogo.com/logo-png/53/1/lyngby-bk-logo-png_seeklogo-536053.png",
        "AaB": "https://aabsport.dk/media/14091/aab_logo_originalt_3000x3000.png?anchor=center&mode=crop&width=1200&height=1200&rnd=132042915090000000",
        "Midtjylland": "https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/FC_Midtjylland_logo.svg/800px-FC_Midtjylland_logo.svg.png",
        "AGF": "https://upload.wikimedia.org/wikipedia/en/a/ac/AGF_Aarhus_logo.svg",
        "Nordsjælland": "https://upload.wikimedia.org/wikipedia/en/thumb/2/23/FC_Nordsj%C3%A6lland_logo.svg/1200px-FC_Nordsj%C3%A6lland_logo.svg.png",
        "Brøndby": "https://w7.pngwing.com/pngs/5/36/png-transparent-br%C3%B8ndby-if-danish-superliga-f-c-copenhagen-danish-cup-fc-nordsj%C3%A6lland-embroidery-danish-cup-symbol-randers-fc-thumbnail.png",
        "Randers": "https://upload.wikimedia.org/wikipedia/en/b/b7/Randers_FC_logo.svg",
        "Silkeborg": "https://www.silkeborgif.com/wp-content/uploads/2021/08/cropped-SIF_logo.png",
        "SønderjyskE": "https://soenderjyskefodbold.dk/wp-content/uploads/2023/06/soenderjyskefodbold-logo.png",
        "København": "https://www.fck.dk/sites/default/files/FCK_Logo.png",
        "Viborg": "https://www.vff.dk/images/Logo/VFF_primaerlogo_cmyk_compr_med.png",
        "Vejle": "https://seeklogo.com/images/V/vejle-boldklub-logo-06F0A27DF5-seeklogo.com.png"
    }

    # Tilføj TeamId kolonne til DataFrame
    contestants_df['TeamId'] = contestants_df['id'].map(id_to_teamid)

    # Tilføj ImageURL kolonne til DataFrame
    contestants_df['ImageURL'] = contestants_df['name'].map(name_to_imageurl)

    return contestants_df

@st.cache_data
def fetch_expected_goals1(match_id, teams_df, contestants_df):
    expectedgoalsevents_url = f"http://api.performfeeds.com/soccerdata/matchexpectedgoals/1vzv4a2oaoik71bcxuaul4qsjs/{match_id}?_rt=b"
    root = fetch_and_parse_xml(expectedgoalsevents_url)
    events = root.findall(".//event")

    # Create a DataFrame for expected goals
    expectedgoals_data = [{
        'event_id': event.get('id'),
        'event_type': event.get('typeId'),
        'period_id': event.get('periodId'),
        'time_min': event.get('timeMin'),
        'time_sec': event.get('timeSec'),
        'contestant_id': event.get('contestantId'),
        'player_id': event.get('playerId'),
        'player_name': event.get('playerName'),
        'outcome': event.get('outcome'),
        'x': event.get('x'),
        'y': event.get('y'),
        'time_stamp': event.get('timeStamp'),
        'last_modified': event.get('lastModified'),
        'assist': event.get('assist')
    } for event in events]

    expectedgoals_df = pd.DataFrame(expectedgoals_data)

    # Extract qualifiers for xG and Goals
    qualifiers_data = []
    goal_events = set()  # To keep track of goal events
    for event in events:
        qualifiers = event.findall('qualifier')
        for qualifier in qualifiers:
            if qualifier.get('qualifierId') == '321':  # xG qualifier ID
                qualifiers_data.append({
                    'event_id': event.get('id'),
                    'xG': float(qualifier.get('value', 0))
                })
            if qualifier.get('qualifierId') == '374':  # Goal qualifier ID
                goal_events.add(event.get('id'))  # Mark this event as a goal

    qualifiers_df = pd.DataFrame(qualifiers_data)

    # Merge xG data with the expected goals DataFrame
    expectedgoals_df = pd.merge(expectedgoals_df, qualifiers_df, on='event_id', how='left')

    # Add Goal column (1 if the event is a goal, 0 otherwise)
    expectedgoals_df['Goal'] = expectedgoals_df['event_id'].isin(goal_events).astype(int)

    # Replace missing xG values and format the xG column
    expectedgoals_df['xG'].fillna(0, inplace=True)
    expectedgoals_df['xG'] = expectedgoals_df['xG'].round(2)

    # Add name column based on contestant_id
    expectedgoals_df['name'] = expectedgoals_df['contestant_id'].map(
        contestants_df.set_index('id')['name']
    )

    return expectedgoals_df

@st.cache_data
def fetch_tournament_schedule():
    tournamentschedule_url = "https://api.performfeeds.com/soccerdata/tournamentschedule/1vzv4a2oaoik71bcxuaul4qsjs?tmcl=1zqurbs9rmwtk30us5y1v1rtg&_rt=b"
    response = requests.get(tournamentschedule_url)
    root = ET.fromstring(response.content)

    matches = []
    for match in root.findall(".//match"):
        match_data = {
            "match_id": match.get("id"),
            "date": match.get("date"),
            "local_date": match.get("localDate"),
            "home_team_name": match.get("homeContestantName"),
            "away_team_name": match.get("awayContestantName")
        }
        match_data['match_display'] = f"{match_data['local_date']}: {match_data['home_team_name']} vs {match_data['away_team_name']}"
        matches.append(match_data)

    return pd.DataFrame(matches)

def fetch_match_stats(match_id, teams_df):
    url2 = f"https://api.performfeeds.com/soccerdata/matchstats/1vzv4a2oaoik71bcxuaul4qsjs/{match_id}?_rt=b&detailed=yes"
    response2 = requests.get(url2)
    root = ET.fromstring(response2.content)

    team_stats = []
    for team in root.findall(".//teamStats/.."):
        team_id = team.find(".//teamOfficial[@type='manager']").get("id")
        
        # Find the corresponding team name from teams_df
        team_name = teams_df.loc[teams_df['TeamId'] == team_id, 'name'].values[0]
        
        for stat in team.find(".//teamStats").findall(".//stat"):
            team_stats.append({
                "TeamId": team_id,
                "TeamName": team_name,
                "type": stat.get("type"),
                "fh": float(stat.get("fh")) if stat.get("fh") else 0,
                "sh": float(stat.get("sh")) if stat.get("sh") else 0,
                "value": float(stat.text) if stat.text else 0
            })

    # Convert team_stats to a DataFrame
    team_stats_df = pd.DataFrame(team_stats)

    # Filter rows that correspond to 'successfulFinalThirdPasses'
    final_third_passes = team_stats_df[team_stats_df['type'] == 'successfulFinalThirdPasses']

    # Group by 'type' to sum the successfulFinalThirdPasses for both teams
    total_final_third_passes = final_third_passes.groupby('type').agg({
        'fh': 'sum',
        'sh': 'sum',
        'value': 'sum'
    }).reset_index()

    # Create a new DataFrame for 'TotalSucFinalThirdPasses'
    total_row = pd.DataFrame({
        'TeamId': ['Both'],  # Placeholder for team ID since it's for both teams
        'TeamName': ['Both Teams'],  # Indicates this is for both teams
        'type': ['TotalSucFinalThirdPasses'],
        'fh': total_final_third_passes['fh'].values,
        'sh': total_final_third_passes['sh'].values,
        'value': total_final_third_passes['value'].values
    })

    # Concatenate the original team_stats with the new row for TotalSucFinalThirdPasses
    team_stats_df = pd.concat([team_stats_df, total_row], ignore_index=True)

    # Extract total successful final third passes for field tilt calculation
    total_fh = total_final_third_passes['fh'].values[0]
    total_sh = total_final_third_passes['sh'].values[0]
    total_value = total_final_third_passes['value'].values[0]

    # Calculate Field Tilt (%) for each team
    field_tilt_stats = final_third_passes.copy()  # Create a copy for field tilt
    field_tilt_stats['type'] = 'Field Tilt (%)'

    # Field Tilt Calculation with formatting to one decimal place
    field_tilt_stats['fh'] = ((final_third_passes['fh'] / total_fh * 100).fillna(0)).round(1)
    field_tilt_stats['sh'] = ((final_third_passes['sh'] / total_sh * 100).fillna(0)).round(1)
    field_tilt_stats['value'] = ((final_third_passes['value'] / total_value * 100).fillna(0)).round(1)

    # Concatenate the original team_stats with the new row for Field Tilt (%)
    team_stats_df = pd.concat([team_stats_df, field_tilt_stats], ignore_index=True)

    return team_stats_df

def fetch_expected_goals(match_id, teams_df):
    url4 = f"http://api.performfeeds.com/soccerdata/matchexpectedgoals/1vzv4a2oaoik71bcxuaul4qsjs/{match_id}?_rt=b"
    response4 = requests.get(url4)
    root = ET.fromstring(response4.content)

    team_stats1 = []
    for team in root.findall(".//teamStats/.."):
        team_id = team.find(".//teamOfficial[@type='manager']").get("id")
        
        # Find det tilsvarende holdnavn fra teams_df
        team_name = teams_df.loc[teams_df['TeamId'] == team_id, 'name'].values[0]
        
        for stat in team.find(".//teamStats").findall(".//stat"):
            # Runde 'value', 'fh', og 'sh' til to decimaler
            team_stats1.append({
                "TeamId": team_id,
                "TeamName": team_name,
                "type": stat.get("type"),
                "fh": round(float(stat.get("fh")) if stat.get("fh") else 0, 2),
                "sh": round(float(stat.get("sh")) if stat.get("sh") else 0, 2),
                "value": round(float(stat.text) if stat.text else 0, 2)
            })

    # Konverter listen til en DataFrame
    team_stats_df = pd.DataFrame(team_stats1)

    # Find 'bigChanceMissed' og 'bigChanceScored' for at beregne 'Big Chances (Scored)'
    big_chance_missed = team_stats_df[team_stats_df['type'] == 'bigChanceMissed'].copy()
    big_chance_scored = team_stats_df[team_stats_df['type'] == 'bigChanceScored'].copy()

    # Merge for at sikre, at vi har både 'bigChanceMissed' og 'bigChanceScored' for hver TeamId
    big_chances = pd.merge(
        big_chance_missed,
        big_chance_scored,
        on=["TeamId", "TeamName"],
        how="outer",
        suffixes=('_missed', '_scored')
    )

    # Opret en ny variabel for 'Big Chances (Scored)'
    big_chances['fh'] = big_chances['fh_missed'].fillna(0) + big_chances['fh_scored'].fillna(0)
    big_chances['sh'] = big_chances['sh_missed'].fillna(0) + big_chances['sh_scored'].fillna(0)
    big_chances['value'] = big_chances['value_missed'].fillna(0) + big_chances['value_scored'].fillna(0)

    # Opret den nye type 'Big Chances (Scored)'
    big_chances['type'] = 'Big Chances'

    # Vælg de relevante kolonner og fjern de midlertidige kolonner
    big_chances = big_chances[['TeamId', 'TeamName', 'type', 'fh', 'sh', 'value']]

    # Tilføj den nye variabel til den originale DataFrame
    team_stats_df = pd.concat([team_stats_df, big_chances], ignore_index=True)

    return team_stats_df

def combine_stats(match_stats, expected_goals):
    # Combine the two datasets
    combined_stats = pd.concat([match_stats, expected_goals])

    # Pivot the combined stats to have TeamNames as columns and types as rows, showing fh, sh, and value
    pivoted_stats = combined_stats.pivot_table(index=["type"], columns="TeamName", values=["fh", "sh", "value"])

    # Beregn Pass Accuracy (%) som accuratePass / totalPass * 100
    if 'accuratePass' in pivoted_stats.index and 'totalPass' in pivoted_stats.index:
        pass_accuracy = (pivoted_stats.loc['accuratePass'] / pivoted_stats.loc['totalPass']) * 100
        pivoted_stats.loc['Pass Accuracy (%)'] = pass_accuracy.round(1)  # Rund til ét decimal

    return pivoted_stats.fillna(0)  # Fill NA values with 0

# Funktion til at hente og parse XML-data
def fetch_and_parse_xml(url):
    response = requests.get(url)
    response.raise_for_status()  # Tjek om anmodningen var vellykket
    return ET.fromstring(response.text)

# Funktion til at hente og returnere possession events
def fetch_possession_events(match_id, contestants_df):
    # Byg URL'en med det specifikke match_id
    possessionevents_url = f"http://api.performfeeds.com/soccerdata/matchevent/1vzv4a2oaoik71bcxuaul4qsjs/{match_id}?_rt=b"
    
    # Hent og parse XML-data
    possessionevents_xml_root = fetch_and_parse_xml(possessionevents_url)
    
    # Ekstraher alle event-noder
    events = possessionevents_xml_root.findall(".//event")
    
    # Opret en DataFrame fra event-attributterne
    possessionevents_data = [{
        'event_id': event.get('id'),
        'event_type': event.get('typeId'),
        'period_id': event.get('periodId'),
        'time_min': event.get('timeMin'),
        'time_sec': event.get('timeSec'),
        'contestant_id': event.get('contestantId'),
        'player_id': event.get('playerId'),
        'player_name': event.get('playerName'),
        'outcome': event.get('outcome'),
        'x': event.get('x'),
        'y': event.get('y'),
        'time_stamp': event.get('timeStamp')
    } for event in events]
    
    # Opret DataFrame
    possessionevents_df = pd.DataFrame(possessionevents_data)
    
    # Funktion til at ekstrahere kvalifikatorer for hvert event
    def extract_qualifiers(event):
        qualifiers = event.findall('qualifier')
        if qualifiers:
            return [{
                'qualifier_id': qualifier.get('qualifierId'),
                'value': qualifier.get('value'),
                'event_id': event.get('id')
            } for qualifier in qualifiers]
        return []
    
    # Ekstraher kvalifikatorer og kombiner dem i en DataFrame
    qualifiers_data = []
    for event in events:
        qualifiers_data.extend(extract_qualifiers(event))
    
    qualifiers_df = pd.DataFrame(qualifiers_data)
    
    # Merge event data with qualifiers data
    possessionevents_df = pd.merge(possessionevents_df, qualifiers_df, on='event_id', how='left')
    
    def import_qualifier_mapping(file_path):
        # Læs CSV-fil med tab-separator, da dine data ser ud til at være tab-separeret
        qualifier_mapping = pd.read_csv(file_path, sep='\t', dtype=str)
        
        # Tjek om kolonnen 'Q_qualifier_id' findes, og omdøb den til 'qualifier_id'
        if 'Q_qualifier_id' in qualifier_mapping.columns:
            qualifier_mapping.rename(columns={'Q_qualifier_id': 'qualifier_id'}, inplace=True)
        
        # Konverter 'qualifier_id' til numerisk datatype
        qualifier_mapping['qualifier_id'] = pd.to_numeric(qualifier_mapping['qualifier_id'], errors='coerce')
        
        return qualifier_mapping
    
    # Import kvalifikator mapping
    qualifier_mapping_path = "/Users/elmedin/Library/CloudStorage/OneDrive-SønderjyskeFodboldAS/Opta Qualifiers.csv"
    qualifier_mapping = import_qualifier_mapping(qualifier_mapping_path)
    
    # Funktion til at tilføje kvalifikatorbeskrivelser
    def add_qualifier(df, qualifier_mapping):
        df['qualifier_id'] = pd.to_numeric(df['qualifier_id'], errors='coerce')
        df = df.merge(qualifier_mapping, on='qualifier_id', how='left')
        return df
    
    # Tilføj kvalifikatorbeskrivelser
    possessionevents_df = add_qualifier(possessionevents_df, qualifier_mapping)
    
    # Funktion til at erstatte NA med 0 i numeriske kolonner og NA med "" i karakterkolonner
    def replace_na_correctly(df):
        df = df.fillna({col: 0 for col in df.select_dtypes(include=np.number).columns})  # Erstat NA med 0 i numeriske kolonner
        df = df.fillna('')  # Erstat NA med tom streng i karakterkolonner
        return df
    
    # Erstat NA værdier
    possessionevents_df = replace_na_correctly(possessionevents_df)
    
    # Konverter qualifier_id til integer og filtrer derefter
    possessionevents_df['qualifier_id'] = possessionevents_df['qualifier_id'].astype('int64')
    
    # Tilføj name kolonne baseret på contestant_id
    possessionevents_df['name'] = possessionevents_df['contestant_id'].map(
        contestants_df.set_index('id')['name']
    )
    
    # Filtrer passes_df for kun at inkludere relevant kvalifikatorer
    passes_df = possessionevents_df[possessionevents_df['qualifier_id'].isin([140, 141])]
    
    # Konverter 'value' kolonnen til numerisk, tving ikke-numeriske værdier til NaN
    filtered_passes = passes_df.copy()
    filtered_passes['value'] = pd.to_numeric(filtered_passes['value'], errors='coerce')
    
    # Pivot tabellen så vi får endX og endY kolonner
    pivot_passes = filtered_passes.pivot_table(
        index=['event_id', 'event_type', 'period_id', 'time_min', 'time_sec', 
               'contestant_id', 'player_id', 'player_name', 'outcome', 'x', 'y', 'time_stamp', 'name'],
        columns='qualifier_id',
        values='value',
        aggfunc='mean'  # Vi bruger 'mean' for at sikre korrekt aggregation, hvis der er flere værdier
    ).reset_index()
    
    # Omdøb kolonnerne til endX og endY
    pivot_passes.rename(columns={140: 'endX', 141: 'endY'}, inplace=True)
    
    return pivot_passes

def get_team_name(team_id, contestants_df):
    team_name = contestants_df[contestants_df['id'] == team_id]['name']
    return team_name.iloc[0] if not team_name.empty else 'Unknown Team'

def plot_passing_map(passing_df, team_name, ax):
    # Filter for completed passes only
    mask_complete = passing_df['outcome'] == 1
    successful_passes_df = passing_df[mask_complete]
    
    # Filter out the NaN values
    successful_passes_df = successful_passes_df.dropna(subset=['x', 'y', 'endX', 'endY'])
    
    # Create a pitch
    pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='white', line_color='black')
    pitch.draw(figsize=(10, 7), ax=ax)
    
    # Bestem farven baseret på team_name
    if team_name == 'Midtjylland':
        line_color = 'black'
        scatter_color = 'black'
    elif team_name == 'SønderjyskE':
        line_color = 'lightblue'
        scatter_color = 'lightblue'
    elif team_name == 'Vejle':
        line_color = 'darkred'
        scatter_color = 'darkred'
    elif team_name == 'AaB':
        line_color = 'red'
        scatter_color = 'red'
    elif team_name == 'København':
        line_color = 'grey'
        scatter_color = 'grey'
    elif team_name == 'Brøndby':
        line_color = 'yellow'
        scatter_color = 'yellow'
    elif team_name == 'AGF':
        line_color = 'white'
        scatter_color = 'white'
    elif team_name == 'Nordsjælland':
        line_color = 'orange'
        scatter_color = 'orange'
    elif team_name == 'Silkeborg':
        line_color = '#77dde7'
        scatter_color = '#77dde7'
    elif team_name == 'Lyngby':
        line_color = 'darkblue'
        scatter_color = 'darkblue'
    elif team_name == 'Randers':
        line_color = '#00008B'
        scatter_color = '#00008B'
    elif team_name == 'Viborg':
        line_color = 'green'
        scatter_color = 'green'
    else:
        line_color = 'gray'  # Default color for other teams
        scatter_color = 'gray'

    # Plot the passes with custom line color and size
    pitch.lines(
        successful_passes_df['x'], successful_passes_df['y'],
        successful_passes_df['endX'], successful_passes_df['endY'],
        lw=5, transparent=True, comet=True, label='Completed Passes',
        color=line_color, ax=ax
    )
    
    # Add edges to end locations with the correct scatter color
    pitch.scatter(
        successful_passes_df['endX'], successful_passes_df['endY'],
        color=scatter_color, s=100, edgecolor='black', zorder=5,
        ax=ax
    )
    
    # Set the title with the team name
    ax.set_title(f'{team_name} - Passes to Final Third', fontsize=20, color='black')
    
    # Make the background transparent
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)

# Function to plot the shot map for each team
def plot_shot_map(shot_data, team_name, ax):
    pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='none', line_color='black')  # Create a vertical pitch
    pitch.draw(ax=ax)  # Draw the pitch on the given axis

    # Ensure that 'x' and 'y' are numeric
    shot_data['x'] = pd.to_numeric(shot_data['x'], errors='coerce')
    shot_data['y'] = pd.to_numeric(shot_data['y'], errors='coerce')

    # Remove any rows with missing values for 'x' or 'y'
    shot_data = shot_data.dropna(subset=['x', 'y'])

    # Bestem farven baseret på team_name
    team_colors = {
        'Midtjylland': 'black', 
        'SønderjyskE': 'lightblue', 
        'Vejle': 'darkred', 
        'AaB': 'red', 
        'København': 'grey', 
        'Brøndby': 'yellow', 
        'AGF': 'white', 
        'Nordsjælland': 'orange', 
        'Silkeborg': '#77dde7', 
        'Lyngby': 'darkblue', 
        'Randers': '#00008B', 
        'Viborg': 'green'
    }
    
    shot_color = team_colors.get(team_name, 'gray')  # Default to gray for other teams

    # Split the shot_data into goals and non-goals
    goals = shot_data[shot_data['Goal'] == 1]
    non_goals = shot_data[shot_data['Goal'] == 0]

    # Plot non-goals with only edges (no fill)
    pitch.scatter(non_goals['x'], non_goals['y'], ax=ax, facecolors='none', edgecolors=shot_color, s=175, linewidth=2, label='Shots')

    # Plot goals with filled circles
    pitch.scatter(goals['x'], goals['y'], ax=ax, c=shot_color, s=175, edgecolor='black', label='Goals')

    # Add a title for the shot map
    ax.set_title(f'{team_name} - Shots', fontsize=20, color='black')

    # Make the background of the figure and axis transparent
    fig = ax.get_figure()
    fig.patch.set_alpha(0.0)  # Make the figure background transparent
    ax.patch.set_alpha(0.0)   # Make the axis background transparent

@st.cache_data
def calculate_cumulative_xg(expectedgoals_df):
    # Filtrér de nødvendige kolonner fra expectedgoals_df
    cumulative_xg_df = expectedgoals_df[['contestant_id', 'time_min', 'xG', 'Goal', 'player_name', 'period_id']].copy()

    # Konverter 'time_min' til numerisk type (i tilfælde af, at den er en streng)
    cumulative_xg_df['time_min'] = pd.to_numeric(cumulative_xg_df['time_min'], errors='coerce')

    # Juster tidspunkterne for 2. halvleg (period_id = 2)
    cumulative_xg_df.loc[cumulative_xg_df['period_id'] == 2, 'time_min'] += 45

    # Sortér data efter 'contestant_id' og 'time_min'
    cumulative_xg_df = cumulative_xg_df.sort_values(by=['contestant_id', 'time_min'])

    # Beregn kumulativ xG for hver 'contestant_id'
    cumulative_xg_df['cumulative_xG'] = cumulative_xg_df.groupby('contestant_id')['xG'].cumsum()

    return cumulative_xg_df

def plot_cumulative_xg(cumulative_xg_df, team_names):
    fig, ax = plt.subplots(figsize=(15, 6))

    # Iterer over alle hold baseret på contestant_id og plot en linje for hver
    for team_id in cumulative_xg_df['contestant_id'].unique():
        team_df = cumulative_xg_df[cumulative_xg_df['contestant_id'] == team_id]
        team_name = team_names[team_id]
        team_xg_total = team_df['cumulative_xG'].max()

        # Bestem farven baseret på team_name
        if team_name == 'Midtjylland':
            line_color = 'black'
            goal_color = 'black'
        elif team_name == 'SønderjyskE':
            line_color = 'lightblue'
            goal_color = 'lightblue'
        elif team_name == 'Vejle':
            line_color = 'darkred'
            goal_color = 'darkred'
        elif team_name == 'AaB':
            line_color = 'red'
            goal_color = 'red'
        elif team_name == 'København':
            line_color = 'grey'
            goal_color = 'grey'
        elif team_name == 'Brøndby':
            line_color = 'yellow'
            goal_color = 'yellow'
        elif team_name == 'AGF':
            line_color = 'white'
            goal_color = 'white'
        elif team_name == 'Nordsjælland':
            line_color = '#FDE809'
            goal_color = '#FDE809'
        elif team_name == 'Silkeborg':
            line_color = '#0098D2'
            goal_color = '#0098D2'
        elif team_name == 'Lyngby':
            line_color = '#023871'
            goal_color = '#023871'
        elif team_name == 'Randers':
            line_color = '#00008B'
            goal_color = '#00008B'
        elif team_name == 'Viborg':
            line_color = '#007662'
            goal_color = '#007662'
        else:
            line_color = 'gray'  # Standardfarve til andre hold
            goal_color = 'gray'

        # Plot kumulativ xG for holdet med tykkere linje og farve baseret på team_name
        ax.step(x=team_df['time_min'], y=team_df['cumulative_xG'], where='post', 
                label=f"{team_name} ({team_xg_total:.2f} xG)", linewidth=2, color=line_color)

        # Highlight de minutter hvor der er scoret mål, og tilføj labels med period_id, time_min, player_name og xG
        goals_df = team_df[team_df['Goal'] == 1]
        if not goals_df.empty:
            ax.scatter(goals_df['time_min'], goals_df['cumulative_xG'], 
                    marker='o', s=120, facecolors='none', edgecolor=goal_color, label=f"{team_name} Goals")

            # Tilføj labels med spillerens navn og xG værdi
            for idx, row in goals_df.iterrows():
                period_label = f"{int(row['period_id'])}H'"
                label_text = f"{period_label} {int(row['time_min'])} - {row['player_name']} ({row['xG']:.2f})"
                ax.annotate(label_text, 
                            (row['time_min'], row['cumulative_xG']),  # Positionen af målet
                            textcoords="offset points",  # Placering af teksten relativt til målet
                            xytext=(0, 40),  # Flyt teksten 150 punkter op (kan justeres efter behov)
                            ha='center',  # Juster horisontalt
                            arrowprops=dict(arrowstyle='-', color='grey', lw=1))  # Tilføj en streg fra teksten til målet

        # Tilpas diagrammet med større tekst
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], fontsize=14)  # Juster 'fontsize' til den ønskede størrelse
        plt.yticks([0, 0.5, 1, 1.5, 2, 2.5], fontsize=15)  # Juster 'fontsize' til den ønskede størrelse
        plt.grid(True)

    # Tilføj grå gridlines
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Gør baggrunden på figuren transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Tilføj tekst og labels i midten
    fig.text(s='Cumulative xG', x=0.5, y=0.97, fontsize=20, ha='center')

    plt.ylabel("Expected Goals (xG)", fontsize=16, labelpad=20)
    plt.xlabel("Minutes", fontsize=16, labelpad=20)
    st.pyplot(fig)

def plot_top_10_xg_shots(expectedgoals_df, home_team_name, away_team_name, top_n=10):
    # Sorter efter xG i faldende rækkefølge og vælg de top 10 skud
    top_shots_df = expectedgoals_df.sort_values(by='xG', ascending=False).head(top_n)

    # Tilføj en tæller, så der ikke er duplikat-etiketter (for fx flere skud i samme minut)
    top_shots_df['shot_label'] = top_shots_df.groupby(['name', 'time_min']).cumcount() + 1
    top_shots_df['label'] = top_shots_df['player_name'] + " (" + top_shots_df['time_min'].astype(str) + "' #" + top_shots_df['shot_label'].astype(str) + ")"

    # Bestem farven baseret på holdets navn
    def get_team_color(team_name):
        if team_name == 'Midtjylland':
            return 'black'
        elif team_name == 'SønderjyskE':
            return 'lightblue'
        elif team_name == 'Vejle':
            return 'darkred'
        elif team_name == 'AaB':
            return 'red'
        elif team_name == 'København':
            return 'grey'
        elif team_name == 'Brøndby':
            return 'yellow'
        elif team_name == 'AGF':
            return 'white'
        elif team_name == 'Nordsjælland':
            return '#FDE809'
        elif team_name == 'Silkeborg':
            return '#0098D2'
        elif team_name == 'Lyngby':
            return '#023871'
        elif team_name == 'Randers':
            return '#00008B'
        elif team_name == 'Viborg':
            return '#007662'
        else:
            return 'gray'  # Default farve

    # Tilføj farve baseret på hold
    top_shots_df['color'] = top_shots_df['name'].apply(get_team_color)

    # Opret plottet
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot de top 10 skud med farver baseret på hold
    bars = ax.barh(top_shots_df['label'], top_shots_df['xG'], color=top_shots_df['color'])

    # Tilføj xG-værdier som labels for enden af barerne
    for bar, xg in zip(bars, top_shots_df['xG']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{xg:.2f}', va='center', ha='left', fontsize=12)

    # Fjern titlen
    ax.set_title("")

    # Gør baggrunden på figuren gennemsigtig/transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Vend y-aksen så de største xG vises øverst
    ax.invert_yaxis()

    # Tilføj labels til akserne
    ax.set_xlabel("xG")

    # Tilpas y-aksens labels (fjern minut og skudtæller fra navnet)
    ax.set_yticklabels([f"{player.split('(')[0]} ({player.split('(')[1].split()[0]}" for player in top_shots_df['label']])

    # Fjern gridlinjer for et mere rent look
    ax.grid(False)

    # Fjern rammerne på figuren for at gøre den mere gennemsigtig
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Vis figuren i Streamlit
    st.pyplot(fig)

# Funktion til at hente og parse XML-data
def fetch_and_parse_xml(url):
    response = requests.get(url)
    response.raise_for_status()  # Tjek om anmodningen var vellykket
    return ET.fromstring(response.text)

# Funktion til at hente og behandle match events baseret på valgt match_id
def fetch_match_events(match_id):
    # URL til match events med dynamisk match_id
    matchevents_url = f"http://api.performfeeds.com/soccerdata/matchevent/1vzv4a2oaoik71bcxuaul4qsjs/{match_id}?_rt=b"
    matchevents_xml_root = fetch_and_parse_xml(matchevents_url)

    # Extract all event nodes
    events = matchevents_xml_root.findall(".//event")

    # Create a DataFrame from the event attributes
    matchevents_data = [{
        'event_id': event.get('id'),
        'event_type': event.get('typeId'),
        'period_id': event.get('periodId'),
        'time_min': event.get('timeMin'),
        'time_sec': event.get('timeSec'),
        'contestant_id': event.get('contestantId'),
        'player_id': event.get('playerId'),
        'player_name': event.get('playerName'),
        'outcome': event.get('outcome'),
        'x': event.get('x'),
        'y': event.get('y'),
        'time_stamp': event.get('timeStamp')
    } for event in events]

    matchevents_df = pd.DataFrame(matchevents_data)

    # Function to extract qualifiers for each event
    def extract_qualifiers(event):
        qualifiers = event.findall('qualifier')
        if qualifiers:
            return [{
                'qualifier_id': qualifier.get('qualifierId'),
                'value': qualifier.get('value'),
                'event_id': event.get('id')
            } for qualifier in qualifiers]
        return []

    # Extract qualifiers and combine into a DataFrame
    qualifiers_data = []
    for event in events:
        qualifiers_data.extend(extract_qualifiers(event))

    qualifiers_df = pd.DataFrame(qualifiers_data)

    # Merge event data with qualifiers data
    matchevents_df = pd.merge(matchevents_df, qualifiers_df, on='event_id', how='left')

    matchevents_df['qualifier_id'] = pd.to_numeric(matchevents_df['qualifier_id'], errors='coerce')
    matchevents_df_filtered = matchevents_df[matchevents_df['qualifier_id'].isin([140, 141])]

    # Pivot matchevents_df to get endX and endY as columns
    pivoted_matchevents_df = matchevents_df_filtered.pivot_table(
        index=['event_id', 'event_type', 'period_id', 'time_min', 'time_sec', 'contestant_id', 'player_id', 'player_name', 'outcome', 'x', 'y'],
        columns='qualifier_id',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Rename the columns for clarity
    pivoted_matchevents_df.rename(columns={140: 'endX', 141: 'endY'}, inplace=True)

    pivoted_matchevents_df = pivoted_matchevents_df[pivoted_matchevents_df['outcome'] == '1']

    xT_grid = pd.read_csv('/Users/elmedin/Library/CloudStorage/OneDrive-SønderjyskeFodboldAS/xT_Grid.csv', header=None)
    xT_grid = np.array(xT_grid)
    xT_rows, xT_cols = xT_grid.shape

    # Opret 'location' baseret på 'x' og 'y' som tuple uden citationstegn
    pivoted_matchevents_df['location'] = list(zip(pivoted_matchevents_df['x'].astype(float), pivoted_matchevents_df['y'].astype(float)))

    # Opret 'end_location' baseret på 'endX' og 'endY' som tuple uden citationstegn
    pivoted_matchevents_df['end_location'] = list(zip(pivoted_matchevents_df['endX'].astype(float), pivoted_matchevents_df['endY'].astype(float)))

    # Ensure x, y, end_x, and end_y columns are numeric, converting non-numeric values to NaN
    pivoted_matchevents_df['x'] = pd.to_numeric(pivoted_matchevents_df['x'], errors='coerce')
    pivoted_matchevents_df['y'] = pd.to_numeric(pivoted_matchevents_df['y'], errors='coerce')
    pivoted_matchevents_df['endX'] = pd.to_numeric(pivoted_matchevents_df['endX'], errors='coerce')
    pivoted_matchevents_df['endY'] = pd.to_numeric(pivoted_matchevents_df['endY'], errors='coerce')

    # Drop any rows with missing or invalid values in x, y, end_x, or end_y
    pivoted_matchevents_df = pivoted_matchevents_df.dropna(subset=['x', 'y', 'endX', 'endY'])

    # Bin the pitch coordinates based on the xT grid dimensions
    pivoted_matchevents_df['start_x_bin'] = pd.cut(pivoted_matchevents_df['x'], bins=xT_cols, labels=False)
    pivoted_matchevents_df['start_y_bin'] = pd.cut(pivoted_matchevents_df['y'], bins=xT_rows, labels=False)
    pivoted_matchevents_df['end_x_bin'] = pd.cut(pivoted_matchevents_df['endX'], bins=xT_cols, labels=False)
    pivoted_matchevents_df['end_y_bin'] = pd.cut(pivoted_matchevents_df['endY'], bins=xT_rows, labels=False)

    # Erstat NaN med 0 for at undgå fejl ved konvertering til heltal
    pivoted_matchevents_df[['start_x_bin', 'start_y_bin', 'end_x_bin', 'end_y_bin']] = pivoted_matchevents_df[['start_x_bin', 'start_y_bin', 'end_x_bin', 'end_y_bin']].fillna(0)

    # Beregn start_zone_value og end_zone_value baseret på xT_grid
    pivoted_matchevents_df['start_zone_value'] = pivoted_matchevents_df[['start_x_bin', 'start_y_bin']].apply(
        lambda z: xT_grid[int(z[1])][int(z[0])], axis=1)

    pivoted_matchevents_df['end_zone_value'] = pivoted_matchevents_df[['end_x_bin', 'end_y_bin']].apply(
        lambda z: xT_grid[int(z[1])][int(z[0])], axis=1)

    # Beregn xT som forskellen mellem start_zone_value og end_zone_value
    pivoted_matchevents_df['xT'] = pivoted_matchevents_df['end_zone_value'] - pivoted_matchevents_df['start_zone_value']

    # **Tilføjet:** Filtrer kun rækker med positive xT-værdier
    pivoted_matchevents_df = pivoted_matchevents_df[pivoted_matchevents_df['xT'] > 0]

    return pivoted_matchevents_df

# Funktion til at bestemme holdfarve baseret på holdets navn
def get_team_color(team_name):
    team_colors = {
        'Midtjylland': 'black',
        'SønderjyskE': 'lightblue',
        'Vejle': 'darkred',
        'AaB': 'red',
        'København': 'grey',
        'Brøndby': 'yellow',
        'AGF': 'white',
        'Nordsjælland': '#FDE809',
        'Silkeborg': '#0098D2',
        'Lyngby': '#023871',
        'Randers': '#00008B',
        'Viborg': '#007662'
    }
    return team_colors.get(team_name, 'gray')  # Default farve

def plot_xt_heatmap(df, team_name, ax):

    # Ensure that 'x' and 'y' columns are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Remove any rows with missing values for 'x', 'y', or 'xT'
    df = df.dropna(subset=['x', 'y', 'xT'])

    # Filter out negative xT values and rows where 'x' is equal to 100
    df = df[(df['xT'] > 0) & (df['x'] != 100)]

    # Create a pitch
    pitch = VerticalPitch(half=True, line_color='#cfcfcf', line_zorder=2, pitch_color='#15242e', pitch_type='opta')
    
    # Draw the pitch on the given axis
    pitch.draw(ax=ax)
    
    # Use 'Blues' colormap for heatmap visualization
    cmap = 'Blues'  # You can replace this with other colormaps like 'plasma', 'coolwarm', etc.
    
    # Create KDE plot with weights from the xT column
    kdeplot = pitch.kdeplot(
        df['x'],
        df['y'],
        ax=ax,
        cmap=cmap,
        fill=True,
        levels=100,
        shade_lowest=False,
        weights=df['xT']
    )

    # Set the background to be transparent
    fig = ax.get_figure()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

def plot_passing_map1(df, team_name, ax):

    # Ensure that 'x', 'y', 'endX', 'endY', and 'xT' columns are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['endX'] = pd.to_numeric(df['endX'], errors='coerce')
    df['endY'] = pd.to_numeric(df['endY'], errors='coerce')
    df['xT'] = pd.to_numeric(df['xT'], errors='coerce')

    # Remove any rows with missing values for 'x', 'y', 'endX', 'endY', or 'xT'
    df = df.dropna(subset=['x', 'y', 'endX', 'endY', 'xT'])

    # Filter to include only passes with positive xT values
    df = df[df['xT'] > 0]

    # Create a pitch
    pitch = VerticalPitch(half=True, line_color='#cfcfcf', line_zorder=2, pitch_color='#15242e', pitch_type='opta')
    
    # Draw the pitch on the given axis
    pitch.draw(ax=ax)

    # Get the team's color using the get_team_color function
    team_color = get_team_color(team_name)

    # Normalize xT values to scale between 0 and 1 for transparency
    norm = plt.Normalize(vmin=df['xT'].min(), vmax=df['xT'].max())

    # Plot the passes with the team's color and varying transparency based on xT values
    for i, row in df.iterrows():
        alpha_value = norm(row['xT'])  # Normalize xT value to use as transparency
        pitch.lines(row['x'], row['y'],
                    row['endX'], row['endY'],
                    lw=5, transparent=True, comet=True, color=team_color, ax=ax, alpha=alpha_value)

    # Set the background to be transparent
    fig = ax.get_figure()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

def plot_momentum_chart(pivoted_matchevents_df, teams_df, contestants_df, home_team_id, away_team_id):
    # Step 1: Calculate max xT per minute for each contestant
    max_xT_per_minute = pivoted_matchevents_df.groupby(['contestant_id', 'time_min'])['xT'].max().reset_index()

    # Step 2: Ensure 'time_min' is numeric and clean
    max_xT_per_minute['time_min'] = pd.to_numeric(max_xT_per_minute['time_min'], errors='coerce')
    max_xT_per_minute = max_xT_per_minute.dropna(subset=['time_min'])
    max_xT_per_minute['time_min'] = max_xT_per_minute['time_min'].astype(int)

    # Step 3: Get unique minutes
    minutes = sorted(max_xT_per_minute['time_min'].unique())

    # Create a dictionary to map contestant_id to team_name
    team_names = {home_team_id: contestants_df.loc[contestants_df['id'] == home_team_id, 'name'].values[0],
                  away_team_id: contestants_df.loc[contestants_df['id'] == away_team_id, 'name'].values[0]}

    # Calculate momentum for each team
    weighted_xT_sum = {home_team_id: [], away_team_id: []}
    momentum = []
    window_size = 2  # Number of minutes to look back
    decay_rate = 0.1  # Exponential decay rate

    # Step 4: Calculate momentum for each minute
    for current_minute in minutes:
        for team in [home_team_id, away_team_id]:
            recent_xT_values = max_xT_per_minute[
                (max_xT_per_minute['contestant_id'] == team) &
                (max_xT_per_minute['time_min'] <= current_minute) &
                (max_xT_per_minute['time_min'] > current_minute - window_size)
            ]

            # Apply exponential decay
            weights = np.exp(-decay_rate * (current_minute - recent_xT_values['time_min'].values))
            weighted_sum = np.sum(weights * recent_xT_values['xT'].values)
            weighted_xT_sum[team].append(weighted_sum)

        # Calculate momentum with home team as positive and away team as negative
        momentum_value = weighted_xT_sum[home_team_id][-1] - weighted_xT_sum[away_team_id][-1]
        momentum.append(momentum_value)

    # Create momentum DataFrame
    momentum_df = pd.DataFrame({
        'minute': minutes,
        'momentum': momentum
    })

    # Plot the momentum bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for home and away teams
    home_team_name = team_names[home_team_id]
    away_team_name = team_names[away_team_id]
    home_team_color = get_team_color(home_team_name)
    away_team_color = get_team_color(away_team_name)

    # Create the bar chart
    bars = ax.bar(momentum_df['minute'], momentum_df['momentum'], color=[home_team_color if m > 0 else away_team_color for m in momentum_df['momentum']], alpha=0.6)

    # Style the plot
    ax.set_title('Momentum Bar Chart', fontsize=16, color='black', pad=20)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_xlim([0, 95])
    ax.set_ylim([-0.3, 0.3])
    ax.set_xlabel('Minute')
    ax.set_ylabel('Momentum')

    # Set background color to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    return fig

def main():
    # Custom CSS for background image and transparent header
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4yjWjnPg/Sydbank-Park-Light-BW.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    # Sideskift-muligheder i sidebaren
    page_choice = st.sidebar.selectbox(
        "Vælg en side",
        ["Match Report", "1st Half"]
    )

    # Inject the custom CSS into the app
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; font-size: 60px;'>MATCH REPORT</h1>", unsafe_allow_html=True)
    
    # Fetch data
    teams_df = fetch_team_data()
    schedule_df = fetch_tournament_schedule()
    contestants_df = fetch_team_data()
    
    # Vælg kamp
    match_choice_display = st.selectbox("Choose match", schedule_df["match_display"].tolist())
    match_choice = schedule_df.loc[schedule_df["match_display"] == match_choice_display, "match_id"].iloc[0]

    if match_choice:

        # Fetch stats for selected match
        match_stats = fetch_match_stats(match_choice, teams_df)
        expected_goals = fetch_expected_goals(match_choice, teams_df)
        combined_stats = combine_stats(match_stats, expected_goals)

        # Omdøb variablerne
        rename_mapping = {
            "expectedGoalsSetplay": "xG Set Pieces",
            "expectedGoalsontarget": "xG on Target",
            "expectedGoals": "xG",
            "ontargetScoringAtt": "On Target",
            "cornerTaken": "Corners",
            "goals": "Goals",
            "touchesInOppBox": "Touches In Box",
            "penAreaEntries": "Box Entries",
            "totalScoringAtt": "Shots (On target)",
            "finalThirdEntries": "Final 3rd Entries",
            "successfulFinalThirdPasses": "Successful Final 3rd Passes",
            "totalFinalThirdPasses": "Final 3rd Passes",
            "possessionPercentage": "Possession (%)",
            "bigChanceMissed": "Big Chances (Missed)",
            "bigChanceScored": "Big Chances (Scored)",
            "totalCrossNocorner": "Crosses (Within 18 yards)",
            "crosses18yard": "Crosses Within 18 yards",
            "possWonAtt3rd": "Possessions Won In Attack 3rd"
        }

        combined_stats.rename(index=rename_mapping, inplace=True)

        # Display team logos for the selected match
        home_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "home_team_name"].iloc[0]
        away_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "away_team_name"].iloc[0]

        # Filter the teams DataFrame to include only the teams playing in the selected match
        selected_teams_df = teams_df[teams_df['name'].isin([home_team_name, away_team_name])]

        # Liste over alle tilgængelige statistiktyper
        all_stats = combined_stats.index.unique().tolist()

        # Foruddefinerede variabler, som skal være valgt som standard
        default_stats = [
        "Goals", "Possession (%)", "On Target", "Shots (On target)", "Big Chances", "xG", "xG on Target", "xG Set Pieces", "Box Entries", "Touches In Box", "Crosses (Within 18 yards)", "Crosses Within 18 yards", "Field Tilt (%)", "Final 3rd Entries", "Pass Accuracy (%)"]

        # Fjern eventuelle standardvariabler, der ikke findes i all_stats
        default_stats = [stat for stat in default_stats if stat in all_stats]

        # Flervalgsmenu til at vælge statistikker - starter med specifikke variabler valgt
        selected_stats = st.multiselect("Choose variables", options=all_stats, default=default_stats)

        # Vælg periode uden tekstlabel, og vis valgmulighederne vandret
        period_choice = st.radio(
            label="",  # Fjern label
            options=['1ST', '2ND', 'FT'],  # Juster valgmulighederne
            horizontal=True  # Gør knapperne vandrette
        )

        # Filtrer statistikkerne baseret på brugerens valg
        if period_choice == '1ST':
            filtered_stats = combined_stats['fh'].loc[selected_stats]
        elif period_choice == '2ND':
            filtered_stats = combined_stats['sh'].loc[selected_stats]
        else:
            filtered_stats = combined_stats['value'].loc[selected_stats]

        # Arrange logos based on home and away team
        home_logo_url = selected_teams_df[selected_teams_df['name'] == home_team_name]['ImageURL'].iloc[0]
        away_logo_url = selected_teams_df[selected_teams_df['name'] == away_team_name]['ImageURL'].iloc[0]

        # Kontrollér om "Goals"-variablen eksisterer, ellers opret den og giv begge hold 0 som værdi
        if "Goals" not in filtered_stats.index:
            filtered_stats.loc["Goals"] = {home_team_name: 0, away_team_name: 0}

        # Get the goals for each team as integers (handle NaN by setting it to 0)
        home_goals = int(filtered_stats.loc["Goals", home_team_name] if "Goals" in filtered_stats.index and not pd.isna(filtered_stats.loc["Goals", home_team_name]) else 0)
        away_goals = int(filtered_stats.loc["Goals", away_team_name] if "Goals" in filtered_stats.index and not pd.isna(filtered_stats.loc["Goals", away_team_name]) else 0)

        # Display the combined stats in a side-by-side format
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>MATCH REPORT</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.image(home_logo_url, width=150)
        with col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 90px;'>{home_goals} - {away_goals}</h1>", unsafe_allow_html=True)
        with col3:
            st.image(away_logo_url, width=150)

        # Display stats with variable names centered and values left and right
        for stat in selected_stats:
            home_stat = filtered_stats.loc[stat, home_team_name] if stat in filtered_stats.index else 0
            away_stat = filtered_stats.loc[stat, away_team_name] if stat in filtered_stats.index else 0
            if stat == "Pass Accuracy (%)":
                home_stat_str = f"{home_stat:.1f}%"
                away_stat_str = f"{away_stat:.1f}%"
            # Håndter specifikke statistikker som "Shots on Target" og "Crosses Within 18 yards"
            elif stat == "Shots (On target)":
                shots_on_target_home = int(filtered_stats.loc["On Target", home_team_name]) if "On Target" in filtered_stats.index and not pd.isna(filtered_stats.loc["On Target", home_team_name]) else 0
                shots_on_target_away = int(filtered_stats.loc["On Target", away_team_name]) if "On Target" in filtered_stats.index and not pd.isna(filtered_stats.loc["On Target", away_team_name]) else 0
                home_stat_str = f"{int(home_stat)} ({shots_on_target_home})"
                away_stat_str = f"{int(away_stat)} ({shots_on_target_away})"
            elif stat == "Crosses (Within 18 yards)":
                crosses_within_18_yards_home = int(filtered_stats.loc["Crosses Within 18 yards", home_team_name]) if "Crosses Within 18 yards" in filtered_stats.index and not pd.isna(filtered_stats.loc["Crosses Within 18 yards", home_team_name]) else 0
                crosses_within_18_yards_away = int(filtered_stats.loc["Crosses Within 18 yards", away_team_name]) if "Crosses Within 18 yards" in filtered_stats.index and not pd.isna(filtered_stats.loc["Crosses Within 18 yards", away_team_name]) else 0
                home_stat_str = f"{int(home_stat)} ({crosses_within_18_yards_home})"
                away_stat_str = f"{int(away_stat)} ({crosses_within_18_yards_away})"
            elif stat == "Crosses Within 18 yards":
                continue  # Skip displaying "Crosses Within 18 yards" as it's already handled with "Shots"
            elif stat == "On Target":
                continue  # Skip displaying "Shots on Target" as it's already handled with "Shots"
            elif stat == "Goals":
                continue  # Skip displaying "Goals" as it's already handled with "Shots"
            else:
                if stat in ["xG", "xG on Target", "xG Set Pieces"]:
                    home_stat_str = f"{home_stat:.2f}"
                    away_stat_str = f"{away_stat:.2f}"
                elif stat == "Possession (%)":
                    home_stat_str = f"{home_stat:.1f}%"
                    away_stat_str = f"{away_stat:.1f}%"
                elif stat == "Field Tilt (%)":
                    home_stat_str = f"{home_stat:.1f}%"
                    away_stat_str = f"{away_stat:.1f}%"
                else:
                    home_stat_str = str(int(home_stat))
                    away_stat_str = str(int(away_stat)) 

            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown(f"<p style='text-align: left; font-size: 24px;'><strong>{home_stat_str}</strong></p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<p style='text-align: center; font-size: 24px;'>{stat}</p>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<p style='text-align: right; font-size: 24px;'><strong>{away_stat_str}</strong></p>", unsafe_allow_html=True)

        # Fetch possession events
        pivot_passes = fetch_possession_events(match_choice, contestants_df)
        
        # Clean data
        pivot_passes['x'] = pd.to_numeric(pivot_passes['x'], errors='coerce')
        pivot_passes['y'] = pd.to_numeric(pivot_passes['y'], errors='coerce')
        pivot_passes['endX'] = pd.to_numeric(pivot_passes['endX'], errors='coerce')
        pivot_passes['endY'] = pd.to_numeric(pivot_passes['endY'], errors='coerce')
        pivot_passes = pivot_passes.dropna(subset=['x', 'y', 'endX', 'endY'])

        # Convert the 'outcome' column to numeric
        pivot_passes['outcome'] = pd.to_numeric(pivot_passes['outcome'], errors='coerce')
        pivot_passes = pivot_passes.dropna(subset=['outcome'])

        # Filter data for completed passes only
        completed_pivot_passes = pivot_passes[pivot_passes['outcome'] == 1]
        
        # Filter data based on the new condition
        filtered_pivot_passes = completed_pivot_passes[(completed_pivot_passes['x'] < 66) & (completed_pivot_passes['endX'] > 66)]

        # Check if there is data to plot
        if not filtered_pivot_passes.empty:
            # Create a dictionary to store traces for each team
            team_traces = {}

            # Create columns for layout
            col1, col2 = st.columns([1, 1])

            for team_id in filtered_pivot_passes['contestant_id'].unique():
                team_df = filtered_pivot_passes[filtered_pivot_passes['contestant_id'] == team_id]
                team_name = get_team_name(team_id, contestants_df)

                fig, ax = plt.subplots(figsize=(10, 7))
                plot_passing_map(team_df, team_name, ax)

                fig.set_facecolor('white')
                fig.patch.set_alpha(0.0)

                if team_name == home_team_name:
                    with col1:
                        st.pyplot(fig)
                elif team_name == away_team_name:
                    with col2:
                        st.pyplot(fig)

        # Fetch expected goals data for the selected match
        expectedgoals_df = fetch_expected_goals1(match_choice, teams_df, contestants_df)

        # Check if there is expected goals data
        if not expectedgoals_df.empty:
            # Create columns for layout to display both shot maps side by side, with home team on the left and away team on the right
            col1, col2 = st.columns([1, 1])

        # Iterate through the teams in the match
            for team_id in expectedgoals_df['contestant_id'].unique():
                team_shots = expectedgoals_df[expectedgoals_df['contestant_id'] == team_id]
                team_name = get_team_name(team_id, contestants_df)

                fig, ax = plt.subplots(figsize=(10, 7))
                plot_shot_map(team_shots, team_name, ax)

                if team_name == home_team_name:
                    with col1:
                        st.pyplot(fig)
                elif team_name == away_team_name:
                    with col2:
                        st.pyplot(fig)

        # Fetch match events baseret på valgt match_id
        pivoted_df = fetch_match_events(match_choice)

        # Get home and away team names and ids
        home_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "home_team_name"].iloc[0]
        away_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "away_team_name"].iloc[0]
        home_team_id = contestants_df[contestants_df['name'] == home_team_name]['id'].values[0]
        away_team_id = contestants_df[contestants_df['name'] == away_team_name]['id'].values[0]

        # Filter data for each team based on their contestant_id
        home_team_df = pivoted_df[pivoted_df['contestant_id'] == home_team_id]
        away_team_df = pivoted_df[pivoted_df['contestant_id'] == away_team_id]

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        # Plot heatmap for hjemmehold in the left column
        with col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 16px; font-weight: normal;'>{home_team_name} - xT Heatmap</h1>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            plot_xt_heatmap(home_team_df, home_team_name, ax1)
            st.pyplot(fig1)

        # Plot heatmap for udehold in the right column
        with col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 16px; font-weight: normal;'>{away_team_name} - xT Heatmap</h1>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            plot_xt_heatmap(away_team_df, away_team_name, ax2)
            st.pyplot(fig2)

        # Fetch match events based on selected match_id
        pivoted_df = fetch_match_events(match_choice)

        # Get home and away team names and ids
        home_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "home_team_name"].iloc[0]
        away_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "away_team_name"].iloc[0]
        home_team_id = contestants_df[contestants_df['name'] == home_team_name]['id'].values[0]
        away_team_id = contestants_df[contestants_df['name'] == away_team_name]['id'].values[0]

        # Filter data for each team based on their contestant_id
        home_team_df = pivoted_df[pivoted_df['contestant_id'] == home_team_id]
        away_team_df = pivoted_df[pivoted_df['contestant_id'] == away_team_id]

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        # Plot passing map for hjemmehold in the left column
        with col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 16px; font-weight: normal;'>{home_team_name} - Expected Threat</h1>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            plot_passing_map1(home_team_df, home_team_name, ax1)
            st.pyplot(fig1)

        # Plot passing map for udehold in the right column
        with col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 16px; font-weight: normal;'>{away_team_name} - Expected Threat</h1>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            plot_passing_map1(away_team_df, away_team_name, ax2)
            st.pyplot(fig2)

        # Beregn kumulativ xG
        cumulative_xg_df = calculate_cumulative_xg(expectedgoals_df)

        # Opret en dictionary for holdnavne baseret på contestant_id
        team_names = {team_id: teams_df.loc[teams_df['id'] == team_id, 'name'].values[0] 
              for team_id in cumulative_xg_df['contestant_id'].unique()}

        # Plot kumulativ xG for alle hold i kampen
        plot_cumulative_xg(cumulative_xg_df, team_names)

        if not expectedgoals_df.empty:
            st.markdown("<h2 style='text-align: center; font-size: 20px; color: black; font-weight: normal;'>Top 10 Shots by xG</h2>", unsafe_allow_html=True)
            plot_top_10_xg_shots(expectedgoals_df, home_team_name, away_team_name, top_n=10)
    
        # Fetch match events and prepare for momentum calculation
        pivoted_df = fetch_match_events(match_choice)

        # Plot momentum chart
        momentum_fig = plot_momentum_chart(pivoted_df, teams_df, contestants_df, home_team_id, away_team_id)

        # Display the momentum chart in Streamlit
        st.pyplot(momentum_fig)

        # Fetch match events based on selected match_id
        pivoted_matchevents_df = fetch_match_events(match_choice)

        # Group by 'contestant_id' and 'player_name' to aggregate the xT values
        player_xt_totals = pivoted_matchevents_df.groupby(['contestant_id', 'player_name'])['xT'].sum().reset_index()

        # Sort each group by xT in descending order and take the top 5 players for each team
        top_5_players_per_team = player_xt_totals.groupby('contestant_id').apply(
            lambda x: x.nlargest(5, 'xT')).reset_index(drop=True)

        # Merge to add team names for display (if needed, based on contestant_id)
        top_5_players_per_team = top_5_players_per_team.merge(
            contestants_df[['id', 'name']],
            left_on='contestant_id',
            right_on='id',
            how='left'
        )

        # Rename columns for clarity
        top_5_players_per_team.rename(columns={'name': 'team_name'}, inplace=True)

        # Get home and away team names
        home_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "home_team_name"].iloc[0]
        away_team_name = schedule_df.loc[schedule_df["match_id"] == match_choice, "away_team_name"].iloc[0]

        # Filter top 5 players for each team and sort by xT descending
        home_team_xt = top_5_players_per_team[top_5_players_per_team['team_name'] == home_team_name].sort_values(by='xT', ascending=True)
        away_team_xt = top_5_players_per_team[top_5_players_per_team['team_name'] == away_team_name].sort_values(by='xT', ascending=True)

        # Get the team colors using the get_team_color function
        home_team_color = get_team_color(home_team_name)
        away_team_color = get_team_color(away_team_name)

        # Create two side-by-side bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot for home team with the team color and transparency
        bars1 = ax1.barh(home_team_xt['player_name'], home_team_xt['xT'], color=home_team_color, alpha=0.7)
        ax1.set_title(f"Expected Threat - {home_team_name}")
        ax1.set_xlabel('xT')
        ax1.set_ylabel('Player')

        # Add xT values at the end of each bar for home team
        for bar in bars1:
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
                    f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)

        # Plot for away team with the team color and transparency
        bars2 = ax2.barh(away_team_xt['player_name'], away_team_xt['xT'], color=away_team_color, alpha=0.7)
        ax2.set_title(f"Expected Threat - {away_team_name}")
        ax2.set_xlabel('xT')
        ax2.set_ylabel('Player')

        # Add xT values at the end of each bar for away team
        for bar in bars2:
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
                    f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)

        # Make background transparent for the plots
        fig.patch.set_alpha(0.0)
        ax1.patch.set_alpha(0.0)
        ax2.patch.set_alpha(0.0)

        # Display the plots in Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()