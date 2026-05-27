import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Page Setup
st.set_page_config(page_title="IPL CRUNCH '26", page_icon="🏏", layout="wide")

# Cache data loading so it runs fast
@st.cache_data
def load_data():
    df = pd.read_csv('data.zip', low_memory=False)
    df = df.dropna(subset=['winner'])
    
    df['toss_won_match'] = df['toss_winner'] == df['winner']
    df['batting_team_won'] = df['batting_team'] == df['winner']
    
    def get_phase(over):
        if over <= 5: return 'Powerplay (1-6)'
        elif over <= 15: return 'Middle Overs (7-16)'
        else: return 'Death Overs (17-20)'
    df['phase'] = df['over'].apply(get_phase)
    
    valid_wicket_kinds = ['caught', 'bowled', 'lbw', 'stumped', 'hit wicket', 'caught and bowled']
    df['is_bowler_wicket'] = df['wicket_kind'].isin(valid_wicket_kinds)
    
    return df

df = load_data()
matches = df.drop_duplicates(subset='match_id')

st.title("IPL CRUNCH '26 — Five Seasons Decoded")
st.markdown("### Find out what actually wins matches using 289,000+ rows of ball-by-ball data.")

# Calculations
toss_win_rate = matches['toss_won_match'].mean() * 100
decision_rates = matches.groupby('toss_decision')['toss_won_match'].mean() * 100
bat_rate = decision_rates.get('bat', 0)
field_rate = decision_rates.get('field', 0)

toss_won = matches['toss_won_match'].sum()
toss_lost = len(matches) - toss_won
chi2, p_value, _, _ = chi2_contingency([[toss_won, toss_lost]])

phase_runs = df.groupby(['match_id', 'batting_team', 'phase', 'batting_team_won'])['runs_total'].sum().reset_index()
phase_balls = df.groupby(['match_id', 'batting_team', 'phase', 'batting_team_won'])['ball'].count().reset_index()
phase_stats = phase_runs.merge(phase_balls, on=['match_id', 'batting_team', 'phase', 'batting_team_won'])
phase_stats['overs'] = phase_stats['ball'] / 6
phase_stats['run_rate'] = phase_stats['runs_total'] / phase_stats['overs']
avg_rr = phase_stats.groupby(['phase', 'batting_team_won'])['run_rate'].mean().reset_index()
pivot_rr = avg_rr.pivot(index='phase', columns='batting_team_won', values='run_rate').reset_index()
pivot_rr.columns = ['phase', 'Losing_Team_RR', 'Winning_Team_RR']
phase_order = ['Powerplay (1-6)', 'Middle Overs (7-16)', 'Death Overs (17-20)']
pivot_rr['phase'] = pd.Categorical(pivot_rr['phase'], categories=phase_order, ordered=True)
pivot_rr = pivot_rr.sort_values('phase')

top_batters = df.groupby('batter').agg(total_runs=('runs_batter', 'sum'), balls_faced=('ball', 'count'), matches=('match_id', 'nunique')).reset_index()
top_batters['strike_rate'] = (top_batters['total_runs'] / top_batters['balls_faced']) * 100
top_batters = top_batters.sort_values('total_runs', ascending=False).head(5)

wickets_df = df[df['is_bowler_wicket'] == True]
top_bowlers = wickets_df.groupby('bowler').agg(total_wickets=('is_bowler_wicket', 'sum'), matches=('match_id', 'nunique')).reset_index()
bowler_runs = df.groupby('bowler').agg(runs_conceded=('runs_total', 'sum'), balls_bowled=('ball', 'count')).reset_index()
top_bowlers = top_bowlers.merge(bowler_runs, on='bowler', how='left')
top_bowlers['economy'] = (top_bowlers['runs_conceded'] / top_bowlers['balls_bowled']) * 6
top_bowlers = top_bowlers.sort_values('total_wickets', ascending=False).head(5)

# Surprising Finding Banner
st.success(f"⚡ SURPRISING FINDING: The toss is a myth (win rate {toss_win_rate:.1f}%, p-value {p_value:.3f}). Matches are won in the Death Overs — winning teams outscore losers by 27.3% in overs 17-20, double the Powerplay gap (12.9%).")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Toss Win Rate")
    fig1, ax1 = plt.subplots()
    bars = ax1.bar(['Toss Winner\nWins Match', 'Toss Winner\nLoses Match'], 
                   [toss_win_rate, 100 - toss_win_rate], 
                   color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
    ax1.set_ylim(0, 70)
    ax1.set_ylabel('Percentage (%)')
    for bar in bars: 
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                 f'{bar.get_height():.1f}%', ha='center', fontweight='bold', fontsize=14)
    ax1.text(0.5, 0.65, f'Field First Win Rate: {field_rate:.1f}%\nBat First Win Rate: {bat_rate:.1f}%', 
             transform=ax1.transAxes, fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    st.pyplot(fig1)

with col2:
    st.subheader("Run Rate Gap by Match Phase")
    fig2, ax2 = plt.subplots()
    max_rr = pivot_rr['Winning_Team_RR'].max()
    for i, row in pivot_rr.iterrows():
        ax2.plot([row['Losing_Team_RR'], row['Winning_Team_RR']], [row['phase'], row['phase']], 
                  color='#bdc3c7', linewidth=6, zorder=1)
        ax2.scatter(row['Winning_Team_RR'], row['phase'], color='#2ecc71', s=300, edgecolor='black', linewidth=1.5, zorder=2)
        ax2.scatter(row['Losing_Team_RR'], row['phase'], color='#e74c3c', s=300, edgecolor='black', linewidth=1.5, zorder=2)
        ax2.text(row['Winning_Team_RR'] + 0.1, row['phase'], f"{row['Winning_Team_RR']:.2f}", fontweight='bold', color='#27ae60', fontsize=10, va='center', ha='left')
        ax2.text(row['Losing_Team_RR'] - 0.1, row['phase'], f"{row['Losing_Team_RR']:.2f}", fontweight='bold', color='#c0392b', fontsize=10, va='center', ha='right')
        gap = ((row['Winning_Team_RR'] - row['Losing_Team_RR']) / row['Losing_Team_RR']) * 100
        ax2.text(max_rr + 2.2, row['phase'], f"+{gap:.1f}%", fontweight='bold', color='#8e44ad', fontsize=11, va='center', ha='center',
                  bbox=dict(facecolor='#f3e5f5', edgecolor='#8e44ad', boxstyle='round,pad=0.4'))
    ax2.scatter([], [], color='#2ecc71', s=150, edgecolor='black', label='Winning Teams')
    ax2.scatter([], [], color='#e74c3c', s=150, edgecolor='black', label='Losing Teams')
    ax2.set_xlabel('Average Run Rate (Per Over)')
    ax2.legend(loc='lower right')
    ax2.set_xlim(pivot_rr['Losing_Team_RR'].min() - 0.8, max_rr + 4.0)
    st.pyplot(fig2)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 5 Run Scorers")
    st.dataframe(top_batters[['batter', 'total_runs', 'matches', 'strike_rate']].style.format({'strike_rate': '{:.1f}'}), use_container_width=True, hide_index=True)

with col4:
    st.subheader("Top 5 Wicket Takers")
    st.dataframe(top_bowlers[['bowler', 'total_wickets', 'matches', 'economy']].style.format({'economy': '{:.1f}'}), use_container_width=True, hide_index=True)
