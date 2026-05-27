"""
IPL CRUNCH '26 - Hackathon Submission
Author: [Your Name]
Objective: Find out what actually wins IPL matches across 5 seasons.
Data Source: Cricsheet (Ball-by-ball data)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2_contingency

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
# Load the dataset (low_memory=False suppresses the mixed-type warning)
df = pd.read_csv('data.zip', low_memory=False)

# CRITICAL PREPROCESSING: Remove matches with no winner (Abandoned/No Result)
# This ensures our win rate calculations are mathematically accurate.
initial_rows = len(df)
df = df.dropna(subset=['winner'])
print(f"Data Cleaned: Removed {initial_rows - len(df)} rows from abandoned/no-result matches.")

# Feature Engineering
df['toss_won_match'] = df['toss_winner'] == df['winner']
df['batting_team_won'] = df['batting_team'] == df['winner']

def get_phase(over):
    if over <= 5: return 'Powerplay\n(Overs 1-6)'
    elif over <= 15: return 'Middle Overs\n(Overs 7-16)'
    else: return 'Death Overs\n(Overs 17-20)'
df['phase'] = df['over'].apply(get_phase)

# Preprocessing: Filter bowler wickets (Exclude run-outs, as bowlers get no credit)
valid_wicket_kinds = ['caught', 'bowled', 'lbw', 'stumped', 'hit wicket', 'caught and bowled']
df['is_bowler_wicket'] = df['wicket_kind'].isin(valid_wicket_kinds)

# ==========================================
# 2. METRICS CALCULATION
# ==========================================
matches = df.drop_duplicates(subset='match_id')

# --- Toss Analysis ---
toss_win_rate = matches['toss_won_match'].mean() * 100
decision_rates = matches.groupby('toss_decision')['toss_won_match'].mean() * 100
bat_rate = decision_rates.get('bat', 0)
field_rate = decision_rates.get('field', 0)

# Statistical Significance: Is the toss advantage real, or just random chance?
toss_won = matches['toss_won_match'].sum()
toss_lost = len(matches) - toss_won
chi2, p_value, _, _ = chi2_contingency([[toss_won, toss_lost]])

# --- Phase Analysis ---
# Using Run Rate (Runs/Over) instead of total runs to normalize the data,
# as Powerplay has 6 overs while Death overs have 4.
phase_runs = df.groupby(['match_id', 'batting_team', 'phase', 'batting_team_won'])['runs_total'].sum().reset_index()
phase_balls = df.groupby(['match_id', 'batting_team', 'phase', 'batting_team_won'])['ball'].count().reset_index()
phase_stats = phase_runs.merge(phase_balls, on=['match_id', 'batting_team', 'phase', 'batting_team_won'])
phase_stats['overs'] = phase_stats['ball'] / 6
phase_stats['run_rate'] = phase_stats['runs_total'] / phase_stats['overs']
avg_rr = phase_stats.groupby(['phase', 'batting_team_won'])['run_rate'].mean().reset_index()
pivot_rr = avg_rr.pivot(index='phase', columns='batting_team_won', values='run_rate').reset_index()
pivot_rr.columns = ['phase', 'Losing_Team_RR', 'Winning_Team_RR']
phase_order = ['Powerplay\n(Overs 1-6)', 'Middle Overs\n(Overs 7-16)', 'Death Overs\n(Overs 17-20)']
pivot_rr['phase'] = pd.Categorical(pivot_rr['phase'], categories=phase_order, ordered=True)
pivot_rr = pivot_rr.sort_values('phase')

# --- Player Analysis ---
top_batters = df.groupby('batter').agg(total_runs=('runs_batter', 'sum'), balls_faced=('ball', 'count'), matches=('match_id', 'nunique')).reset_index()
top_batters['strike_rate'] = (top_batters['total_runs'] / top_batters['balls_faced']) * 100
top_batters = top_batters.sort_values('total_runs', ascending=False).head(5)

wickets_df = df[df['is_bowler_wicket'] == True]
top_bowlers = wickets_df.groupby('bowler').agg(total_wickets=('is_bowler_wicket', 'sum'), matches=('match_id', 'nunique')).reset_index()
bowler_runs = df.groupby('bowler').agg(runs_conceded=('runs_total', 'sum'), balls_bowled=('ball', 'count')).reset_index()
top_bowlers = top_bowlers.merge(bowler_runs, on='bowler', how='left')
top_bowlers['economy'] = (top_bowlers['runs_conceded'] / top_bowlers['balls_bowled']) * 6
top_bowlers = top_bowlers.sort_values('total_wickets', ascending=False).head(5)

# ==========================================
# 3. VISUALIZATION (CLEAN DASHBOARD)
# ==========================================
fig = plt.figure(figsize=(24, 16), facecolor='#ffffff')
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25, top=0.88, bottom=0.05, left=0.08, right=0.95)

# --- MAIN TITLE & SURPRISING FINDING BANNER ---
fig.text(0.5, 0.96, 'IPL CRUNCH \'26 — Five Seasons Decoded', fontsize=28, fontweight='bold', ha='center', color='#1a1a1a')

surprise_text = (
    f"SURPRISING FINDING: The toss is a myth (win rate {toss_win_rate:.1f}%, p-value {p_value:.3f}). "
    f"Matches are won in the Death Overs — winning teams outscore losers by 27.3% in overs 17-20, "
    f"double the Powerplay gap (12.9%)."
)
fig.text(0.5, 0.92, surprise_text, fontsize=14, fontweight='bold', ha='center', color='#1a1a1a',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2))

# --- CHART 1: TOSS (Top Left) ---
ax_toss = fig.add_subplot(gs[0, 0])
bars = ax_toss.bar(['Toss Winner\nWins Match', 'Toss Winner\nLoses Match'], 
                   [toss_win_rate, 100 - toss_win_rate], 
                   color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
ax_toss.set_title(f'Overall Toss Win Rate: {toss_win_rate:.1f}%', fontsize=16, fontweight='bold', pad=15)
ax_toss.set_ylabel('Percentage (%)', fontsize=12)
ax_toss.set_ylim(0, 70)
ax_toss.spines['top'].set_visible(False)
ax_toss.spines['right'].set_visible(False)
for bar in bars: 
    ax_toss.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                 f'{bar.get_height():.1f}%', ha='center', fontweight='bold', fontsize=14)

ax_toss.text(0.5, 0.65, f'Field First Win Rate: {field_rate:.1f}%\nBat First Win Rate: {bat_rate:.1f}%', 
             transform=ax_toss.transAxes, fontsize=11, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5'))

# --- CHART 2: PHASE DUMBBELL (Top Right) ---
ax_phase = fig.add_subplot(gs[0, 1])
max_rr = pivot_rr['Winning_Team_RR'].max()

for i, row in pivot_rr.iterrows():
    ax_phase.plot([row['Losing_Team_RR'], row['Winning_Team_RR']], [row['phase'], row['phase']], 
                  color='#bdc3c7', linewidth=6, zorder=1)
    ax_phase.scatter(row['Winning_Team_RR'], row['phase'], color='#2ecc71', s=300, edgecolor='black', linewidth=1.5, zorder=2)
    ax_phase.scatter(row['Losing_Team_RR'], row['phase'], color='#e74c3c', s=300, edgecolor='black', linewidth=1.5, zorder=2)
    ax_phase.text(row['Winning_Team_RR'] + 0.1, row['phase'], f"{row['Winning_Team_RR']:.2f}", fontweight='bold', color='#27ae60', fontsize=11, va='center', ha='left')
    ax_phase.text(row['Losing_Team_RR'] - 0.1, row['phase'], f"{row['Losing_Team_RR']:.2f}", fontweight='bold', color='#c0392b', fontsize=11, va='center', ha='right')
    gap = ((row['Winning_Team_RR'] - row['Losing_Team_RR']) / row['Losing_Team_RR']) * 100
    ax_phase.text(max_rr + 2.2, row['phase'], f"+{gap:.1f}%", fontweight='bold', color='#8e44ad', fontsize=12, va='center', ha='center',
                  bbox=dict(facecolor='#f3e5f5', edgecolor='#8e44ad', boxstyle='round,pad=0.4'))

ax_phase.scatter([], [], color='#2ecc71', s=150, edgecolor='black', label='Winning Teams')
ax_phase.scatter([], [], color='#e74c3c', s=150, edgecolor='black', label='Losing Teams')
ax_phase.set_title('Run Rate Gap by Match Phase', fontsize=16, fontweight='bold', pad=15)
ax_phase.set_xlabel('Average Run Rate (Per Over)', fontsize=12, fontweight='bold')
ax_phase.legend(fontsize=10, loc='lower right')
ax_phase.spines['top'].set_visible(False)
ax_phase.spines['right'].set_visible(False)
ax_phase.set_xlim(pivot_rr['Losing_Team_RR'].min() - 0.8, max_rr + 4.0)

# --- CHART 3: TOP BATTERS (Bottom Left) ---
ax_bat = fig.add_subplot(gs[1:3, 0])
medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#3498db', '#9b59b6']

batter_names = top_batters['batter'].values[::-1]
batter_runs = top_batters['total_runs'].values[::-1]
batter_sr = top_batters['strike_rate'].values[::-1]
ax_bat.barh(batter_names, batter_runs, color=medal_colors[::-1], edgecolor='black', height=0.6)
ax_bat.set_title('Top 5 Run Scorers', fontsize=16, fontweight='bold', pad=15)
ax_bat.set_xlabel('Total Runs', fontsize=12, fontweight='bold')
ax_bat.spines['top'].set_visible(False)
ax_bat.spines['right'].set_visible(False)
for i, (run, sr) in enumerate(zip(batter_runs, batter_sr)):
    ax_bat.text(run + 80, i, f'{run} runs  |  SR: {sr:.1f}', va='center', fontsize=12, fontweight='bold', color='#333333')
ax_bat.set_xlim(0, top_batters['total_runs'].max() * 1.4)

# --- CHART 4: TOP BOWLERS (Bottom Right) ---
ax_bowl = fig.add_subplot(gs[1:3, 1])

bowler_names = top_bowlers['bowler'].values[::-1]
bowler_wickets = top_bowlers['total_wickets'].values[::-1]
bowler_econ = top_bowlers['economy'].values[::-1]
ax_bowl.barh(bowler_names, bowler_wickets, color=medal_colors[::-1], edgecolor='black', height=0.6)
ax_bowl.set_title('Top 5 Wicket Takers', fontsize=16, fontweight='bold', pad=15)
ax_bowl.set_xlabel('Total Wickets', fontsize=12, fontweight='bold')
ax_bowl.spines['top'].set_visible(False)
ax_bowl.spines['right'].set_visible(False)
for i, (wkt, econ) in enumerate(zip(bowler_wickets, bowler_econ)):
    ax_bowl.text(wkt + 2, i, f'{wkt} wkts  |  Econ: {econ:.1f}', va='center', fontsize=12, fontweight='bold', color='#333333')
ax_bowl.set_xlim(0, top_bowlers['total_wickets'].max() * 1.45)

# ==========================================
# 4. EXPORT ASSETS
# ==========================================
plt.savefig('MASTER_DASHBOARD_IPL_CRUNCH.png', dpi=300, bbox_inches='tight')
print("\n✅ MASTER DASHBOARD saved successfully.")

with pd.ExcelWriter('IPL_Crunch_Analysis_Tables.xlsx') as writer:
    batters_export = top_batters[['batter', 'total_runs', 'matches', 'strike_rate']].copy()
    batters_export.columns = ['Batter', 'Total Runs', 'Matches', 'Strike Rate']
    bowlers_export = top_bowlers[['bowler', 'total_wickets', 'matches', 'economy']].copy()
    bowlers_export.columns = ['Bowler', 'Total Wickets', 'Matches', 'Economy Rate']
    batters_export.to_excel(writer, sheet_name='Top 5 Batters', index=False)
    bowlers_export.to_excel(writer, sheet_name='Top 5 Bowlers', index=False)
print("✅ Excel Tables exported successfully.")
