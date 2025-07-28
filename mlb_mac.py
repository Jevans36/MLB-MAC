import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from io import BytesIO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.interpolate import griddata
import base64

# Memory-efficient imports - only import when needed
@st.cache_resource
def get_sklearn_components():
    """Import sklearn components only when needed"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics.pairwise import euclidean_distances
    from kneed import KneeLocator
    return StandardScaler, GaussianMixture, euclidean_distances, KneeLocator



# Configure Streamlit
st.set_page_config(
    page_title="MAC Baseball Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants - EXACT SAME as MAC_module
color_dict = {"Fastball": "red", "Breaking": "blue", "Offspeed": "green", "Unknown": "gray"}
distance_threshold = 0.6
strike_zone = {"top": 3.3775, "bottom": 1.5, "left": -0.83083, "right": 0.83083}
swing_calls = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]

# === EXACT SAME wOBA weights as MAC_module ===
woba_weights = {
    'Walk': 0.692,
    'HitByPitch': 0.723,
    'Single': 0.885,
    'Double': 1.257,
    'Triple': 1.593,
    'HomeRun': 2.053
}

def clean_numeric_column(series):
    """Convert a series to numeric, replacing non-numeric values with NaN - EXACT SAME as MAC_module"""
    return pd.to_numeric(series, errors='coerce')

class DatabaseManager:
    def __init__(self, db_path="baseball_data.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Download and create database if it doesn't exist"""
        if not os.path.exists(self.db_path):
            st.info("📂 Setting up database for first time use...")
            self.create_database_from_dropbox()
    
    def create_database_from_dropbox(self):
        """Create database from both 2024 and 2025 MLB data (both from Dropbox)"""
        try:
            progress_bar = st.progress(0)
            st.info("📂 Downloading 2024 data from Dropbox...")
            
            # Download 2024 data
            ncaa_url = "https://www.dropbox.com/scl/fi/54o93jhavlk1dlictfhup/statcast_2024.parquet?rlkey=4kud2iumjqb96sh4lb3girjdw&st=4gxqkom7&dl=1"
            response = requests.get(ncaa_url, timeout=300)
            response.raise_for_status()
            progress_bar.progress(30)
            
            ncaa_df = pd.read_parquet(BytesIO(response.content))
            st.success(f"✅ 2024 MLB data loaded: {len(ncaa_df):,} rows")
            progress_bar.progress(50)
            
            # Download CCBL data (if available)
            st.info("📂 Downloading 2025 MLB data from Dropbox...")
            try:
                ccbl_url = "https://www.dropbox.com/scl/fi/guwqimo1k39ivraj5widj/statcast_2025.parquet?rlkey=0afxm2kgtelcw1egs0owkumjx&st=l61m4283&dl=1"
                
                ccbl_response = requests.get(ccbl_url, timeout=180)
                ccbl_response.raise_for_status()
                
                ccbl_df = pd.read_parquet(BytesIO(ccbl_response.content))
                st.success(f"✅ 2025 MLB data loaded: {len(ccbl_df):,} rows")
                df = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
                st.success(f"✅ Combined dataset: {len(df):,} rows")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load CCBL data: {e}")
                st.info("Using 2024 data only")
                df = ncaa_df
            
            progress_bar.progress(70)

            
            
            # Create SQLite database
            conn = sqlite3.connect(self.db_path)
            df.to_sql('pitches', conn, if_exists='replace', index=False)
            progress_bar.progress(85)
            
            # Create indexes
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX idx_pitcher ON pitches(Pitcher)")
            cursor.execute("CREATE INDEX idx_batter ON pitches(Batter)")
            cursor.execute("CREATE INDEX idx_pitcher_batter ON pitches(Pitcher, Batter)")
            
            # Create summary tables
            cursor.execute("""
                CREATE TABLE pitcher_summary AS
                SELECT 
                    Pitcher,
                    COUNT(*) as total_pitches,
                    AVG(RelSpeed) as avg_speed,
                    AVG(InducedVertBreak) as avg_ivb,
                    AVG(HorzBreak) as avg_hb,
                    AVG(SpinRate) as avg_spin
                FROM pitches 
                WHERE RelSpeed IS NOT NULL AND InducedVertBreak IS NOT NULL 
                  AND HorzBreak IS NOT NULL AND SpinRate IS NOT NULL
                GROUP BY Pitcher
                HAVING COUNT(*) >= 10
            """)
            
            conn.commit()
            conn.close()
            progress_bar.progress(100)
            st.success("✅ Database created successfully!")
            
            # Clean up memory
            del ncaa_df
            if 'ccbl_df' in locals():
                del ccbl_df
            del df
            
        except Exception as e:
            st.error(f"❌ Error creating database: {e}")
            raise
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_pitchers(self):
        """Get list of pitchers with sufficient data"""
        conn = self.get_connection()
        query = "SELECT Pitcher FROM pitcher_summary ORDER BY Pitcher"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['Pitcher'].tolist()
    
    def get_batters(self):
        """Get list of batters"""
        conn = self.get_connection()
        query = "SELECT DISTINCT Batter FROM pitches WHERE Batter IS NOT NULL ORDER BY Batter"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['Batter'].tolist()
    
    def get_analysis_data(self, pitcher_name, target_hitters):
        """Get data for analysis - chunked approach for memory efficiency"""
        conn = self.get_connection()
        
        # Get data in smaller chunks
        placeholders = ','.join(['?' for _ in target_hitters])
        
        query = f"""
            SELECT * FROM pitches 
            WHERE (Pitcher = ? OR Batter IN ({placeholders}))
              AND RelSpeed IS NOT NULL 
              AND InducedVertBreak IS NOT NULL 
              AND HorzBreak IS NOT NULL 
              AND SpinRate IS NOT NULL
              AND Pitcher IS NOT NULL 
              AND Batter IS NOT NULL
        """
        
        params = [pitcher_name] + target_hitters
        
        # Try chunked reading, fall back to regular if not supported
        try:
            # Read in chunks to manage memory
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_sql_query(query, conn, params=params, chunksize=chunk_size):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        except TypeError:
            # chunksize not supported in some pandas versions, read all at once
            df = pd.read_sql_query(query, conn, params=params)
        
        conn.close()
        return df

def run_complete_mac_analysis(pitcher_name, target_hitters, db_manager):
    """Complete MAC analysis with ALL original logic preserved"""
    
    st.info("🔧 **MAC Analysis Pipeline Started**")
    
    # === STEP 1: Get Data (EXACT SAME as MAC_module) ===
    with st.spinner("📊 Loading analysis data..."):
        df = db_manager.get_analysis_data(pitcher_name, target_hitters)
        
        if df.empty:
            st.error("No data found in dataset")
            return None, None, None
        
        # Filter for pitcher's data only for clustering (EXACT SAME)
        pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
        if pitcher_pitches.empty:
            st.error(f"No pitches found for pitcher: {pitcher_name}")
            return None, None, None
    
    st.success(f"✅ Data loaded: {len(df):,} total rows, {len(pitcher_pitches):,} pitcher rows")
    
    # === STEP 2: Clean Numeric Columns (EXACT SAME as MAC_module) ===
    with st.spinner("🧹 Cleaning numeric columns..."):
        numeric_columns = [
            'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
            'run_value', 'RunsScored', 'OutsOnPlay', 'ExitSpeed', 'Angle', 'PlateLocHeight', 'PlateLocSide'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col])
        
        # Check for required columns (EXACT SAME)
        required_cols = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide', 'TaggedPitchType']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None, None, None
    
    st.success("✅ Numeric columns cleaned and validated")
    
    # USING CCBL R/OUT AS PREDEFINED - previously was FINDING r/out AFTER filtering for batter/pitcher matchup (invalid approach!)
    LEAGUE_R_OUT = 0.193
    
    # Replace STEP 3 in run_complete_mac_analysis with this:
    def get_league_environment():
        """Return pre-calculated league environment"""
        st.info(f"📈 Using pre-calculated league environment: R/Out = {LEAGUE_R_OUT:.3f}")
        return LEAGUE_R_OUT

    r_out = get_league_environment()
    
    # === STEP 4: Assign wOBA result values (EXACT SAME) ===
    with st.spinner("💯 Assigning wOBA values..."):
        if 'wOBA_result' not in df.columns:
            df['wOBA_result'] = 0.0  # Initialize
            df.loc[df['KorBB'] == 'Walk', 'wOBA_result'] = woba_weights['Walk']
            df.loc[df['PitchCall'] == 'HitByPitch', 'wOBA_result'] = woba_weights['HitByPitch']
            df.loc[df['PlayResult'] == 'Single', 'wOBA_result'] = woba_weights['Single']
            df.loc[df['PlayResult'] == 'Double', 'wOBA_result'] = woba_weights['Double']
            df.loc[df['PlayResult'] == 'Triple', 'wOBA_result'] = woba_weights['Triple']
            df.loc[df['PlayResult'] == 'HomeRun', 'wOBA_result'] = woba_weights['HomeRun']
        else:
            df['wOBA_result'] = clean_numeric_column(df['wOBA_result'])
    
    st.success("✅ wOBA values assigned")
    
    # === STEP 5: Feature sets (EXACT SAME) ===
    scanning_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide']
    clustering_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']
    
    df = df.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    pitcher_pitches = pitcher_pitches.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    
    st.info(f"🎯 Using clustering features: {clustering_features}")
    st.info(f"🔍 Using scanning features: {scanning_features}")
    
    # === STEP 6: Scale features and cluster pitcher's arsenal (EXACT SAME) ===
    with st.spinner("🤖 Clustering pitcher's arsenal..."):
        StandardScaler, GaussianMixture, euclidean_distances, KneeLocator = get_sklearn_components()
        
        # Step 1: Fit using clustering features on pitcher's data (EXACT SAME)
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])
        
        # Step 2: Run BIC loop to find optimal number of clusters (EXACT SAME)
        bic_scores = []
        ks = range(1, 10)
        for k in ks:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_cluster)
            bic_scores.append(gmm.bic(X_cluster))
        
        # Step 3: Find the "elbow" (knee point) (EXACT SAME)
        knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
        optimal_k = knee.elbow or 2  # fallback to 2 if no elbow found
        
        # Step 4: Fit final GMM using optimal_k and assign cluster labels (EXACT SAME)
        best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)
    
    st.success(f"✅ Optimal clusters found: {optimal_k} clusters")
    
    # === STEP 7: Assign PitchGroup using TaggedPitchType majority (EXACT SAME) ===
    with st.spinner("🏷️ Assigning pitch groups..."):
        autopitchtype_to_group = {
            'Four-Seam': 'Fastball',
            'Fastball': 'Fastball',
            'FourSeamFastBall': 'Fastball',
            'TwoSeamFastBall': 'Fastball',
            'Sinker': 'Fastball',
            'Slider': 'Breaking',
            'Cutter': 'Breaking',
            'Curveball': 'Breaking',
            'Sweeper': 'Breaking',
            'Changeup': 'Offspeed',
            'Splitter': 'Offspeed',
            'ChangeUp': 'Offspeed'
        }
        
        # Handle missing TaggedPitchType if any (EXACT SAME)
        pitcher_pitches = pitcher_pitches.dropna(subset=["TaggedPitchType"])
        
        # Compute most common TaggedPitchType for each cluster (EXACT SAME)
        cluster_to_type = {}
        for cluster in pitcher_pitches['PitchCluster'].unique():
            cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
            type_counts = cluster_data['TaggedPitchType'].value_counts()
            
            if type_counts.empty:
                cluster_to_type[cluster] = 'Unknown'
                continue
            
            most_common_type = type_counts.idxmax()
            pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
            cluster_to_type[cluster] = pitch_group
        
        pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
        
        # Compute pitch group usage (EXACT SAME)
        pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()
    
    st.success(f"✅ Pitch groups assigned: {list(pitch_group_usage.keys())}")
    
    # Show pitch group usage
    usage_text = ", ".join([f"{group}: {usage*100:.1f}%" for group, usage in pitch_group_usage.items()])
    st.info(f"🎯 Pitcher's arsenal usage: {usage_text}")
    
    # === STEP 8: Tag FULL dataset with MinDistToPitcher (EXACT SAME CRITICAL FIX) ===
    with st.spinner("📏 Calculating pitch similarity distances..."):
        scaler_all = StandardScaler()
        df_scaled = scaler_all.fit_transform(df[scanning_features])  # FULL dataset
        X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
        distances = euclidean_distances(df_scaled, X_pitcher_full)
        df['MinDistToPitcher'] = distances.min(axis=1)
        
        # Assign PitchGroup to entire dataset using cluster model (EXACT SAME)
        df_subset_scaled = scaler.transform(df[clustering_features])
        df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
        df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)
    
    similar_pitches_count = (df['MinDistToPitcher'] <= distance_threshold).sum()
    st.success(f"✅ Similarity calculated: {similar_pitches_count:,} similar pitches found (threshold: {distance_threshold})")
    
    # === STEP 9: Matchup scoring (EXACT SAME LOGIC) ===
    with st.spinner("🥊 Running matchup analysis..."):
        results = []
        group_breakdown = []
        
        for hitter in target_hitters:
            hitter_result = {"Batter": hitter}
            weighted_stats = []
            total_weight = 0
            
            # Initialize accumulators for summary (EXACT SAME)
            total_pitches_seen = 0
            total_swings_seen = 0
            total_whiffs_seen = 0
            total_ev_sum = 0
            total_la_sum = 0
            total_hard_hits = 0
            total_gbs = 0
            total_bips = 0
            total_hits = 0
            total_outs = 0
            total_woba_num = 0
            total_woba_den = 0
            
            for group, usage in pitch_group_usage.items():
                # NOW using full dataset for matchup analysis (EXACT SAME)
                group_pitches = df[
                    (df["Batter"] == hitter) &
                    (df["PitchGroup"] == group) &
                    (df["MinDistToPitcher"] <= distance_threshold)
                ].copy()
                
                if group_pitches.empty:
                    continue
                
                # Clean plate location columns for zone calculation (EXACT SAME)
                group_pitches['PlateLocHeight'] = clean_numeric_column(group_pitches['PlateLocHeight'])
                group_pitches['PlateLocSide'] = clean_numeric_column(group_pitches['PlateLocSide'])
                
                group_pitches["InZone"] = (
                    (group_pitches["PlateLocHeight"] >= strike_zone["bottom"]) &
                    (group_pitches["PlateLocHeight"] <= strike_zone["top"]) &
                    (group_pitches["PlateLocSide"] >= strike_zone["left"]) &
                    (group_pitches["PlateLocSide"] <= strike_zone["right"])
                )
                group_pitches["Swung"] = group_pitches["PitchCall"].isin(swing_calls)
                group_pitches["Whiff"] = group_pitches["PitchCall"] == "StrikeSwinging"
                group_pitches["IsInPlay"] = group_pitches['PitchCall'].isin(["InPlay"])
                
                total_pitches = len(group_pitches)
                total_swings = group_pitches["Swung"].sum()
                total_whiffs = group_pitches["Whiff"].sum()
                total_run_value = group_pitches["run_value"].sum() if 'run_value' in group_pitches.columns else 0
                
                # Clean exit speed and angle columns (EXACT SAME)
                group_pitches['ExitSpeed'] = clean_numeric_column(group_pitches['ExitSpeed'])
                group_pitches['Angle'] = clean_numeric_column(group_pitches['Angle'])
                
                exit_velo = group_pitches["ExitSpeed"].mean()
                launch_angle = group_pitches["Angle"].mean()
                
                balls_in_play = group_pitches[group_pitches["IsInPlay"]]
                num_ground_balls = (balls_in_play["Angle"] < 10).sum()
                gb_percent = round(100 * num_ground_balls / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan
                num_hard_hits = (balls_in_play["ExitSpeed"] >= 95).sum()
                hh_percent = round(100 * num_hard_hits / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan
                
                rv_per_100 = 100 * total_run_value / total_pitches if total_pitches > 0 else 0
                weighted_stats.append(usage * rv_per_100)
                total_weight += usage
                
                # Calculate AVG for this group (EXACT SAME)
                hit_mask = (
                    (group_pitches["PitchCall"] == "InPlay") &
                    (group_pitches["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]))
                )
                hits = hit_mask.sum()
                
                out_mask = (
                    ((group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                    ((group_pitches["PitchCall"] == "InPlay") & (group_pitches["PlayResult"] == "Out"))) &
                    (group_pitches["PlayResult"] != "Sacrifice")
                )
                outs = out_mask.sum()
                
                avg = round(hits / (hits + outs), 3) if (hits + outs) > 0 else np.nan
                
                # Accumulate full pitch data for summary (EXACT SAME)
                total_pitches_seen += total_pitches
                total_swings_seen += total_swings
                total_whiffs_seen += total_whiffs
                total_bips += len(balls_in_play)
                total_hits += hits
                total_outs += outs
                if not np.isnan(exit_velo):
                    total_ev_sum += exit_velo * len(balls_in_play)
                if not np.isnan(launch_angle):
                    total_la_sum += launch_angle * len(balls_in_play)
                if not np.isnan(num_hard_hits):
                    total_hard_hits += num_hard_hits
                if not np.isnan(num_ground_balls):
                    total_gbs += num_ground_balls
                
                # Compute wOBA for this group (EXACT SAME)
                plate_ending = group_pitches[
                    (group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                    (group_pitches["PitchCall"].isin(["InPlay", "HitByPitch"]))
                ]
                
                group_woba_numerator = plate_ending["wOBA_result"].sum()
                group_woba_denominator = len(plate_ending)
                group_woba = round(group_woba_numerator / group_woba_denominator, 3) if group_woba_denominator > 0 else np.nan
                
                # Accumulate for summary-level wOBA (EXACT SAME)
                total_woba_num += group_woba_numerator
                total_woba_den += group_woba_denominator
                
                group_breakdown.append({
                    "Batter": hitter,
                    "PitchGroup": group,
                    "AVG": avg,
                    "RV/100": round(rv_per_100, 2),
                    "Whiff%": round(100 * total_whiffs / total_swings, 1) if total_swings > 0 else np.nan,
                    "SwStr%": round(100 * total_whiffs / total_pitches, 1) if total_pitches > 0 else np.nan,
                    "HH%": hh_percent,
                    "ExitVelo": round(exit_velo, 1) if not np.isnan(exit_velo) else np.nan,
                    "Launch": round(launch_angle, 1) if not np.isnan(launch_angle) else np.nan,
                    "GB%": gb_percent,
                    "UsageWeight": round(usage, 2),
                    "Pitches": total_pitches,
                    "InPlay": len(balls_in_play),
                    "wOBA": group_woba,
                })
            
            # Summary calculations (EXACT SAME)
            weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else np.nan
            hitter_result["RV/100"] = round(weighted_rv, 2)
            hitter_result["AVG"] = round(total_hits / (total_hits + total_outs), 3) if (total_hits + total_outs) > 0 else np.nan
            hitter_result["Whiff%"] = round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else np.nan
            hitter_result["SwStr%"] = round(100 * total_whiffs_seen / total_pitches_seen, 1) if total_pitches_seen > 0 else np.nan
            hitter_result["ExitVelo"] = round(total_ev_sum / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["Launch"] = round(total_la_sum / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["HH%"] = round(100 * total_hard_hits / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["GB%"] = round(100 * total_gbs / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["Pitches"] = total_pitches_seen
            hitter_result["InPlay"] = total_bips
            hitter_result["wOBA"] = round(total_woba_num / total_woba_den, 3) if total_woba_den > 0 else np.nan
            
            results.append(hitter_result)
    
    st.success("✅ **MAC Analysis Complete!** All original logic preserved")
    
    return pd.DataFrame(results), pd.DataFrame(group_breakdown), df

def run_silent_mac_analysis(pitcher_name, target_hitters, db_manager):
    """Silent MAC analysis - no verbose output for Hot Arms batch processing"""
    
    # === STEP 1: Get Data ===
    df = db_manager.get_analysis_data(pitcher_name, target_hitters)
    
    if df.empty:
        return None, None, None
    
    # Filter for pitcher's data only for clustering
    pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
    if pitcher_pitches.empty:
        return None, None, None
    
    # === STEP 2: Clean Numeric Columns ===
    numeric_columns = [
        'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
        'run_value', 'RunsScored', 'OutsOnPlay', 'ExitSpeed', 'Angle', 'PlateLocHeight', 'PlateLocSide'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Check for required columns
    required_cols = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide', 'TaggedPitchType']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, None, None
    
    # === STEP 3: League Environment ===
    LEAGUE_R_OUT = 0.193
    
    # === STEP 4: Assign wOBA result values ===
    if 'wOBA_result' not in df.columns:
        df['wOBA_result'] = 0.0
        df.loc[df['KorBB'] == 'Walk', 'wOBA_result'] = woba_weights['Walk']
        df.loc[df['PitchCall'] == 'HitByPitch', 'wOBA_result'] = woba_weights['HitByPitch']
        df.loc[df['PlayResult'] == 'Single', 'wOBA_result'] = woba_weights['Single']
        df.loc[df['PlayResult'] == 'Double', 'wOBA_result'] = woba_weights['Double']
        df.loc[df['PlayResult'] == 'Triple', 'wOBA_result'] = woba_weights['Triple']
        df.loc[df['PlayResult'] == 'HomeRun', 'wOBA_result'] = woba_weights['HomeRun']
    else:
        df['wOBA_result'] = clean_numeric_column(df['wOBA_result'])
    
    # === STEP 5: Feature sets ===
    scanning_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide']
    clustering_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']
    
    df = df.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    pitcher_pitches = pitcher_pitches.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    
    # === STEP 6: Scale features and cluster pitcher's arsenal ===
    StandardScaler, GaussianMixture, euclidean_distances, KneeLocator = get_sklearn_components()
    
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])
    
    # BIC loop to find optimal number of clusters
    bic_scores = []
    ks = range(1, 10)
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_cluster)
        bic_scores.append(gmm.bic(X_cluster))
    
    # Find the "elbow" (knee point)
    knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
    optimal_k = knee.elbow or 2
    
    # Fit final GMM and assign cluster labels
    best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)
    
    # === STEP 7: Assign PitchGroup using TaggedPitchType majority ===
    autopitchtype_to_group = {
        'Four-Seam': 'Fastball', 'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball',
        'TwoSeamFastBall': 'Fastball', 'Sinker': 'Fastball', 'Slider': 'Breaking',
        'Cutter': 'Breaking', 'Curveball': 'Breaking', 'Sweeper': 'Breaking',
        'Changeup': 'Offspeed', 'Splitter': 'Offspeed', 'ChangeUp': 'Offspeed'
    }
    
    pitcher_pitches = pitcher_pitches.dropna(subset=["TaggedPitchType"])
    
    cluster_to_type = {}
    for cluster in pitcher_pitches['PitchCluster'].unique():
        cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
        type_counts = cluster_data['TaggedPitchType'].value_counts()
        
        if type_counts.empty:
            cluster_to_type[cluster] = 'Unknown'
            continue
        
        most_common_type = type_counts.idxmax()
        pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
        cluster_to_type[cluster] = pitch_group
    
    pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
    pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()
    
    # === STEP 8: Tag FULL dataset with MinDistToPitcher ===
    scaler_all = StandardScaler()
    df_scaled = scaler_all.fit_transform(df[scanning_features])
    X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
    distances = euclidean_distances(df_scaled, X_pitcher_full)
    df['MinDistToPitcher'] = distances.min(axis=1)
    
    # Assign PitchGroup to entire dataset using cluster model
    df_subset_scaled = scaler.transform(df[clustering_features])
    df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
    df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)
    
    # === STEP 9: Matchup scoring ===
    results = []
    group_breakdown = []
    
    for hitter in target_hitters:
        hitter_result = {"Batter": hitter}
        weighted_stats = []
        total_weight = 0
        
        # Initialize accumulators for summary
        total_pitches_seen = 0
        total_swings_seen = 0
        total_whiffs_seen = 0
        total_ev_sum = 0
        total_la_sum = 0
        total_hard_hits = 0
        total_gbs = 0
        total_bips = 0
        total_hits = 0
        total_outs = 0
        total_woba_num = 0
        total_woba_den = 0
        
        for group, usage in pitch_group_usage.items():
            group_pitches = df[
                (df["Batter"] == hitter) &
                (df["PitchGroup"] == group) &
                (df["MinDistToPitcher"] <= distance_threshold)
            ].copy()
            
            if group_pitches.empty:
                continue
            
            # Clean plate location columns for zone calculation
            group_pitches['PlateLocHeight'] = clean_numeric_column(group_pitches['PlateLocHeight'])
            group_pitches['PlateLocSide'] = clean_numeric_column(group_pitches['PlateLocSide'])
            
            group_pitches["InZone"] = (
                (group_pitches["PlateLocHeight"] >= strike_zone["bottom"]) &
                (group_pitches["PlateLocHeight"] <= strike_zone["top"]) &
                (group_pitches["PlateLocSide"] >= strike_zone["left"]) &
                (group_pitches["PlateLocSide"] <= strike_zone["right"])
            )
            group_pitches["Swung"] = group_pitches["PitchCall"].isin(swing_calls)
            group_pitches["Whiff"] = group_pitches["PitchCall"] == "StrikeSwinging"
            group_pitches["IsInPlay"] = group_pitches['PitchCall'].isin(["InPlay"])
            
            total_pitches = len(group_pitches)
            total_swings = group_pitches["Swung"].sum()
            total_whiffs = group_pitches["Whiff"].sum()
            total_run_value = group_pitches["run_value"].sum() if 'run_value' in group_pitches.columns else 0
            
            # Clean exit speed and angle columns
            group_pitches['ExitSpeed'] = clean_numeric_column(group_pitches['ExitSpeed'])
            group_pitches['Angle'] = clean_numeric_column(group_pitches['Angle'])
            
            exit_velo = group_pitches["ExitSpeed"].mean()
            launch_angle = group_pitches["Angle"].mean()
            
            balls_in_play = group_pitches[group_pitches["IsInPlay"]]
            num_ground_balls = (balls_in_play["Angle"] < 10).sum()
            gb_percent = round(100 * num_ground_balls / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan
            num_hard_hits = (balls_in_play["ExitSpeed"] >= 95).sum()
            hh_percent = round(100 * num_hard_hits / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan
            
            rv_per_100 = 100 * total_run_value / total_pitches if total_pitches > 0 else 0
            weighted_stats.append(usage * rv_per_100)
            total_weight += usage
            
            # Calculate AVG for this group
            hit_mask = (
                (group_pitches["PitchCall"] == "InPlay") &
                (group_pitches["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]))
            )
            hits = hit_mask.sum()
            
            out_mask = (
                ((group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                ((group_pitches["PitchCall"] == "InPlay") & (group_pitches["PlayResult"] == "Out"))) &
                (group_pitches["PlayResult"] != "Sacrifice")
            )
            outs = out_mask.sum()
            
            avg = round(hits / (hits + outs), 3) if (hits + outs) > 0 else np.nan
            
            # Accumulate full pitch data for summary
            total_pitches_seen += total_pitches
            total_swings_seen += total_swings
            total_whiffs_seen += total_whiffs
            total_bips += len(balls_in_play)
            total_hits += hits
            total_outs += outs
            if not np.isnan(exit_velo):
                total_ev_sum += exit_velo * len(balls_in_play)
            if not np.isnan(launch_angle):
                total_la_sum += launch_angle * len(balls_in_play)
            if not np.isnan(num_hard_hits):
                total_hard_hits += num_hard_hits
            if not np.isnan(num_ground_balls):
                total_gbs += num_ground_balls
            
            # Compute wOBA for this group
            plate_ending = group_pitches[
                (group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                (group_pitches["PitchCall"].isin(["InPlay", "HitByPitch"]))
            ]
            
            group_woba_numerator = plate_ending["wOBA_result"].sum()
            group_woba_denominator = len(plate_ending)
            group_woba = round(group_woba_numerator / group_woba_denominator, 3) if group_woba_denominator > 0 else np.nan
            
            # Accumulate for summary-level wOBA
            total_woba_num += group_woba_numerator
            total_woba_den += group_woba_denominator
            
            group_breakdown.append({
                "Batter": hitter, "PitchGroup": group, "AVG": avg, "RV/100": round(rv_per_100, 2),
                "Whiff%": round(100 * total_whiffs / total_swings, 1) if total_swings > 0 else np.nan,
                "SwStr%": round(100 * total_whiffs / total_pitches, 1) if total_pitches > 0 else np.nan,
                "HH%": hh_percent, "ExitVelo": round(exit_velo, 1) if not np.isnan(exit_velo) else np.nan,
                "Launch": round(launch_angle, 1) if not np.isnan(launch_angle) else np.nan,
                "GB%": gb_percent, "UsageWeight": round(usage, 2), "Pitches": total_pitches,
                "InPlay": len(balls_in_play), "wOBA": group_woba,
            })
        
        # Summary calculations
        weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else np.nan
        hitter_result["RV/100"] = round(weighted_rv, 2)
        hitter_result["AVG"] = round(total_hits / (total_hits + total_outs), 3) if (total_hits + total_outs) > 0 else np.nan
        hitter_result["Whiff%"] = round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else np.nan
        hitter_result["SwStr%"] = round(100 * total_whiffs_seen / total_pitches_seen, 1) if total_pitches_seen > 0 else np.nan
        hitter_result["ExitVelo"] = round(total_ev_sum / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["Launch"] = round(total_la_sum / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["HH%"] = round(100 * total_hard_hits / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["GB%"] = round(100 * total_gbs / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["Pitches"] = total_pitches_seen
        hitter_result["InPlay"] = total_bips
        hitter_result["wOBA"] = round(total_woba_num / total_woba_den, 3) if total_woba_den > 0 else np.nan
        
        results.append(hitter_result)
    
    return pd.DataFrame(results), pd.DataFrame(group_breakdown), df


def compute_heatmap_stats(df, metric_col, min_samples=3):
    """Compute heatmap statistics for zone analysis"""
    valid = df[["PlateLocSide", "PlateLocHeight", metric_col]].dropna()
    if len(valid) < min_samples:
        return None, None, None

    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(0.5, 4.5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    try:
        points = valid[["PlateLocSide", "PlateLocHeight"]].values
        values = valid[metric_col].values
        Z = griddata(points, values, (X, Y), method='linear', fill_value=0)

        if len(valid) < 10:
            sigma = 0.5
        elif len(valid) < 25:
            sigma = 1.0
        else:
            sigma = 1.5

        Z_smooth = ndimage.gaussian_filter(Z, sigma=sigma, mode='constant', cval=0)

        mask = np.zeros_like(Z_smooth)
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                dist = np.sqrt((points[:, 0] - x_range[i])**2 + (points[:, 1] - y_range[j])**2)
                if np.min(dist) < 0.8:
                    mask[j, i] = 1

        Z_smooth *= mask
        Z_smooth[Z_smooth < 0.01] = 0
        return x_range, y_range, Z_smooth
    except Exception as e:
        st.error(f"Heatmap error: {e}")
        return None, None, None

def generate_zone_heatmap(df, selected_hitter):
    """Generate zone-level heatmap for a specific hitter"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    metrics = [("WhiffFlag", "Whiff Rate"), ("HardHitFlag", "Hard Hit Rate"), ("wOBA_result", "wOBA")]
    pitch_groups = ["Fastball", "Breaking", "Offspeed"]

    # Add flags to dataframe
    df["WhiffFlag"] = (df["PitchCall"] == "StrikeSwinging").astype(int)
    df["HardHitFlag"] = ((df["ExitSpeed"] >= 95) & df["ExitSpeed"].notna()).astype(int)

    for i, (metric, title) in enumerate(metrics):
        for j, group in enumerate(pitch_groups):
            ax = axes[i, j]
            subset = df[df["PitchGroup"] == group].copy()

            if len(subset) == 0:
                ax.text(0, 2.75, "No Data", ha='center', va='center', fontsize=12)
            else:
                x_range, y_range, z = compute_heatmap_stats(subset, metric)

                if z is not None and np.any(z > 0):
                    X, Y = np.meshgrid(x_range, y_range)

                    if metric == "wOBA_result":
                        actual_max = np.max(z[z > 0])
                        vmax = min(actual_max * 1.1, 1.8)
                        vmin = 0
                        cmap = "RdYlBu_r"
                        levels = np.linspace(vmin, vmax, 20)
                    else:
                        vmin, vmax = 0, 1
                        cmap = "RdYlBu_r"
                        levels = np.linspace(0, 1, 20)

                    cs = ax.contourf(X, Y, z, levels=levels, cmap=cmap,
                                     vmin=vmin, vmax=vmax, alpha=0.8, extend='both')
                    ax.contour(X, Y, z, levels=levels[::4], colors='white',
                               linewidths=0.5, alpha=0.3)

                    if j == 2:
                        cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
                        cbar.set_label(title, rotation=270, labelpad=15)
                else:
                    valid = subset[["PlateLocSide", "PlateLocHeight", metric]].dropna()
                    if not valid.empty:
                        if metric in ["WhiffFlag", "HardHitFlag"]:
                            colors = ['lightblue' if x == 0 else 'red' for x in valid[metric]]
                            ax.scatter(valid["PlateLocSide"], valid["PlateLocHeight"],
                                       c=colors, s=40, alpha=0.7, edgecolors='black')
                        else:
                            vmin, vmax = 0, 1 if metric != "wOBA_result" else min(valid[metric].max() * 1.1, 1.8)
                            sc = ax.scatter(valid["PlateLocSide"], valid["PlateLocHeight"],
                                            c=valid[metric], cmap="RdYlBu_r", s=60,
                                            edgecolors="black", alpha=0.8,
                                            vmin=0, vmax=vmax)
                            if j == 2:
                                fig.colorbar(sc, ax=ax, shrink=0.6)

            # Add strike zone
            strike_zone_rect = patches.Rectangle((-0.83, 1.5), 1.66, 1.8775, linewidth=2.5,
                                               edgecolor='black', facecolor='none')
            ax.add_patch(strike_zone_rect)

            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([1.0, 4.0])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f8f9fa")

            if j == 0:
                ax.set_ylabel(title, fontsize=12, fontweight='bold')
            if i == 2:
                ax.set_xlabel(group, fontsize=12, fontweight='bold')

    fig.suptitle(f"Zone-Level Heat Maps for {selected_hitter}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Convert to base64 for Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def create_comprehensive_visualization(summary_df, breakdown_df, pitcher_name):
    """Create comprehensive visualization matching the Dash app style with percentile lines"""
    fig = make_subplots(rows=1, cols=1)
    
    # Calculate 25th and 75th percentiles from summary data
    if not summary_df.empty and 'RV/100' in summary_df.columns:
        percentile_25 = summary_df['RV/100'].quantile(0.25)
        percentile_75 = summary_df['RV/100'].quantile(0.75)
    else:
        # Fallback values if no data
        percentile_25 = -2
        percentile_75 = 2
    
    # Add summary points with comprehensive hover info
    for _, row in summary_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Batter"]],
            y=[row["RV/100"]],
            mode="markers",
            marker=dict(size=20, color="black"),
            name="Overall",
            hovertemplate=(
                f"<b>{row['Batter']}</b><br>"
                f"RV/100: {row['RV/100']}<br>"
                f"wOBA: {row['wOBA']}<br>"
                f"AVG: {row['AVG']}<br>"
                f"Whiff%: {row['Whiff%']}<br>"
                f"SwStr%: {row['SwStr%']}<br>"
                f"HH%: {row['HH%']}<br>"
                f"GB%: {row['GB%']}<br>"
                f"ExitVelo: {row['ExitVelo']}<br>"
                f"Launch: {row['Launch']}<br>"
                f"Pitches: {row['Pitches']}<br>"
                f"InPlay: {row['InPlay']}<extra></extra>"
            ),
            showlegend=False
        ))
    
    # Add breakdown points with comprehensive hover info
    for _, row in breakdown_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Batter"]],
            y=[row["RV/100"]],
            mode="markers+text",
            marker=dict(size=14, color=color_dict.get(row["PitchGroup"], "gray")),
            text=[f"{int(row['Pitches'])}P"],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertemplate=(
                f"<b>{row['Batter']}</b><br>"
                f"PitchGroup: {row['PitchGroup']}<br>"
                f"RV/100: {row['RV/100']}<br>"
                f"wOBA: {row['wOBA']}<br>"
                f"AVG: {row['AVG']}<br>"
                f"Whiff%: {row['Whiff%']}<br>"
                f"SwStr%: {row['SwStr%']}<br>"
                f"HH%: {row['HH%']}<br>"
                f"GB%: {row['GB%']}<br>"
                f"ExitVelo: {row['ExitVelo']}<br>"
                f"Launch: {row['Launch']}<br>"
                f"Pitches: {row['Pitches']}<br>"
                f"InPlay: {row['InPlay']}<extra></extra>"
            ),
            showlegend=False
        ))
    
    # Add horizontal dashed lines for 25th and 75th percentiles
    # 25th percentile = lower RV/100 = worse for hitters = green
    fig.add_hline(
        y=percentile_25,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"25th Percentile (RV/100: {percentile_25:.2f})",
        annotation_position="bottom right"
    )
    
    # 75th percentile = higher RV/100 = better for hitters = red
    fig.add_hline(
        y=percentile_75,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"75th Percentile (RV/100: {percentile_75:.2f})",
        annotation_position="top right"
    )
    
    fig.update_layout(
        height=700,
        title=f"Expected Matchup RV/100 + Hitter Summary: {pitcher_name} - - - Note that the black indicates expected weighted performance and the red, blue, and green dots represent fastballs, breaking balls, and offspeed, respectively",
        yaxis_title="Better for Pitchers <------    RV/100   ------> Better for Hitters",
        template="simple_white",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(summary_df))),
            ticktext=summary_df["Batter"].tolist(),
            tickangle=45
        )
    )
    
    # Add annotations for better/worse indicators positioned like y-axis labels
    fig.add_annotation(
        xref="paper", yref="y",
        x=-0.1, y=summary_df["RV/100"].max() + 1 if not summary_df.empty else 5,
        text="Better for Hitters",
        showarrow=False,
        font=dict(size=14, color="red"),
        align="center",
        textangle=90  # Rotate text vertically like y-axis label
    )
    
    fig.add_annotation(
        xref="paper", yref="y",
        x=-0.1, y=summary_df["RV/100"].min() - 1 if not summary_df.empty else -5,
        text="Worse for Hitters",
        showarrow=False,
        font=dict(size=14, color="green"),
        align="center",
        textangle=90  # Rotate text vertically like y-axis label
    )
    
    return fig

def create_movement_chart(movement_df):
    """Create pitch movement chart matching Dash app style"""
    movement_df_filtered = movement_df[
        (movement_df["HorzBreak"].between(-50, 50)) & 
        (movement_df["InducedVertBreak"].between(-50, 50))
    ]
    
    fig = go.Figure()
    
    for pitch_type, color in color_dict.items():
        pitch_df = movement_df_filtered[movement_df_filtered["PitchGroup"] == pitch_type]
        if not pitch_df.empty:
            fig.add_trace(go.Scatter(
                x=pitch_df["HorzBreak"],
                y=pitch_df["InducedVertBreak"],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.7),
                name=pitch_type,
                customdata=pitch_df[["Batter", "RelSpeed", "SpinRate"]],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "HB: %{x}<br>"
                              "IVB: %{y}<br>"
                              "RelSpeed: %{customdata[1]} mph<br>"
                              "SpinRate: %{customdata[2]}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Pitch Movement (HorzBreak vs. InducedVertBreak)",
        xaxis=dict(title="Horizontal Break", range=[-30, 30]),
        yaxis=dict(title="Induced Vertical Break", range=[-30, 30], scaleanchor="x", scaleratio=1),
        template="simple_white",
        height=600,
        width=1000
    )
    
    return fig
    # Move these functions OUTSIDE of main() - place them after create_movement_chart()

# Update the analyze_hot_arms_strategy function to use the silent version:
def analyze_hot_arms_strategy(hot_arms, selected_hitters, db_manager):
    """Analyze strategic matchups for available pitchers with minimal output"""
    if not hot_arms or not selected_hitters:
        return None, None
    
    # Show clean progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store results for all pitcher-hitter combinations
    all_matchups = []
    pitcher_summaries = {}
    
    for i, pitcher in enumerate(hot_arms):
        try:
            # Update progress
            progress = (i + 1) / len(hot_arms)
            progress_bar.progress(progress)
            status_text.text(f"🔥 Analyzing {pitcher} ({i+1}/{len(hot_arms)})")
            
            # Run SILENT MAC analysis for this pitcher
            summary_df, breakdown_df, _ = run_silent_mac_analysis(
                pitcher, selected_hitters, db_manager
            )
            
            if summary_df is not None and not summary_df.empty:
                # Store individual matchup data
                for _, row in summary_df.iterrows():
                    all_matchups.append({
                        'Pitcher': pitcher,
                        'Hitter': row['Batter'],
                        'RV/100': row['RV/100'],
                        'wOBA': row['wOBA'],
                        'Pitches': row['Pitches'],
                        'Whiff%': row['Whiff%'],
                        'HH%': row['HH%']
                    })
                
                # Store pitcher summary
                pitcher_summaries[pitcher] = {
                    'avg_rv': summary_df['RV/100'].mean(),
                    'best_matchup': summary_df.loc[summary_df['RV/100'].idxmin(), 'Batter'],
                    'worst_matchup': summary_df.loc[summary_df['RV/100'].idxmax(), 'Batter'],
                    'best_rv': summary_df['RV/100'].min(),
                    'worst_rv': summary_df['RV/100'].max(),
                    'total_pitches': summary_df['Pitches'].sum()
                }
                
        except Exception as e:
            st.warning(f"Could not analyze {pitcher}: {str(e)}")
            continue
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_matchups), pitcher_summaries

def create_matchup_rankings_table(matchups_df, hitter_order):
    """Create color-coded pitcher matchup rankings with preserved hitter order"""
    if matchups_df.empty:
        return None
    
    # Create pivot table for better visualization
    pivot_df = matchups_df.pivot(index='Pitcher', columns='Hitter', values='RV/100')
    
    # Reorder columns to match the original hitter input order
    # Only include hitters that actually exist in the data
    available_hitters = [h for h in hitter_order if h in pivot_df.columns]
    pivot_df = pivot_df[available_hitters]
    
    # Create styled dataframe
    def color_rv_values(val):
        if pd.isna(val):
            return 'background-color: lightgray'
        elif val < -2:
            return 'background-color: darkgreen; color: white'  # Excellent for pitcher
        elif val < 0:
            return 'background-color: lightgreen'  # Good for pitcher
        elif val < 2:
            return 'background-color: lightyellow'  # Neutral
        elif val < 5:
            return 'background-color: lightcoral'  # Bad for pitcher
        else:
            return 'background-color: darkred; color: white'  # Very bad for pitcher
    
    styled_df = pivot_df.style.applymap(color_rv_values).format(precision=2)
    return styled_df

def create_optimal_usage_recommendations(matchups_df, pitcher_summaries):
    """Generate strategic usage recommendations"""
    if matchups_df.empty:
        return []
    
    recommendations = []
    
    # Best overall matchup
    best_overall = matchups_df.loc[matchups_df['RV/100'].idxmin()]
    recommendations.append({
        'type': '🎯 Best Overall Matchup',
        'recommendation': f"{best_overall['Pitcher']} vs {best_overall['Hitter']}",
        'details': f"RV/100: {best_overall['RV/100']:.2f} (Excellent pitcher advantage)"
    })
    
    # Worst matchup to avoid
    worst_overall = matchups_df.loc[matchups_df['RV/100'].idxmax()]
    recommendations.append({
        'type': '⚠️ Matchup to Avoid',
        'recommendation': f"{worst_overall['Pitcher']} vs {worst_overall['Hitter']}",
        'details': f"RV/100: {worst_overall['RV/100']:.2f} (Strong hitter advantage)"
    })
    
    # Best pitcher for each hitter
    for hitter in matchups_df['Hitter'].unique():
        hitter_matchups = matchups_df[matchups_df['Hitter'] == hitter]
        best_pitcher = hitter_matchups.loc[hitter_matchups['RV/100'].idxmin()]
        recommendations.append({
            'type': f'👤 Best vs {hitter}',
            'recommendation': f"{best_pitcher['Pitcher']}",
            'details': f"RV/100: {best_pitcher['RV/100']:.2f}"
        })
    
    # Lineup entry recommendations
    rv_threshold = matchups_df['RV/100'].median()
    
    for pitcher in pitcher_summaries.keys():
        pitcher_matchups = matchups_df[matchups_df['Pitcher'] == pitcher]
        good_matchups = pitcher_matchups[pitcher_matchups['RV/100'] < rv_threshold]
        
        if len(good_matchups) >= 2:
            hitters_list = ", ".join(good_matchups['Hitter'].tolist())
            recommendations.append({
                'type': f'📍 {pitcher} Entry Spots',
                'recommendation': f"Optimal vs: {hitters_list}",
                'details': f"Avg RV/100: {good_matchups['RV/100'].mean():.2f}"
            })
    
    return recommendations

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

def main():
    st.title("⚾ MAC Baseball Analytics")
    st.markdown("**Complete MAC Implementation** - All original logic preserved with step-by-step transparency")
    
    # Initialize database
    try:
        db_manager = get_database_manager()
    except Exception as e:
        st.error(f"❌ Could not initialize database: {e}")
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("🗄️ Database Info")
        if os.path.exists("baseball_data.db"):
            db_size = os.path.getsize("baseball_data.db") / 1024**2
            st.metric("Database Size", f"{db_size:.1f}MB")
            st.success("✅ Database ready")
        
        if st.button("🔄 Refresh Database"):
            if os.path.exists("baseball_data.db"):
                os.remove("baseball_data.db")
            st.cache_resource.clear()
            st.rerun()
        
        st.header("🎯 MAC Algorithm Steps")
        st.markdown("""
        **EXACT SAME as MAC_module:**
        1. **Load & Clean Data** - Clean numeric columns
        2. **wOBA Assignment** - Apply league weights
        3. **Clustering** - GMM + BIC + KneeLocator
        4. **Pitch Grouping** - TaggedPitchType majority
        5. **Similarity** - Euclidean distance calculation
        6. **Matchup Analysis** - Usage-weighted statistics
        """)
        
        st.header("📊 Analysis Parameters")
        st.metric("Distance Threshold", f"{distance_threshold}")
        st.metric("Scanning Features", "6 features")
        st.metric("Clustering Features", "4 features")
    
    # Get available options
    with st.spinner("Loading available players..."):
        try:
            available_pitchers = db_manager.get_pitchers()
            available_batters = db_manager.get_batters()
        except Exception as e:
            st.error(f"❌ Error loading players: {e}")
            st.stop()
    
    # Display stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Pitchers", len(available_pitchers))
    with col2:
        st.metric("Available Batters", len(available_batters))
    
    st.markdown("---")
    
    # Selection interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Pitcher")
        selected_pitcher = st.selectbox(
            "Choose a pitcher:",
            available_pitchers,
            index=0 if available_pitchers else None
        )
    
    with col2:
        st.subheader("Select Hitters")
        selected_hitters = st.multiselect(
            "Choose hitters:",
            available_batters,
            default=available_batters[:3] if len(available_batters) >= 3 else available_batters[:1]
        )
        
        # NEW: Hot Arms selection
        st.subheader("🔥 Hot Arms Available")
        hot_arms = st.multiselect(
            "Select available pitchers for game strategy:",
            available_pitchers,
            default=[],
            help="Choose pitchers available to pitch in this game for strategic analysis"
        )
    
    # Analysis
    # Analysis button - ONLY runs analysis and stores data
    if st.button("🚀 Run Complete MAC Analysis", type="primary", use_container_width=True):
        if not selected_pitcher or not selected_hitters:
            st.warning("Please select both a pitcher and at least one hitter.")
        else:
            st.markdown("---")
            st.header("🔬 MAC Analysis Pipeline")
            
            try:
                summary_df, breakdown_df, full_df = run_complete_mac_analysis(
                    selected_pitcher, selected_hitters, db_manager
                )
                
                if summary_df is not None and not summary_df.empty:
                    # Store results in session state for persistence
                    st.session_state.summary_df = summary_df
                    st.session_state.breakdown_df = breakdown_df
                    st.session_state.selected_pitcher = selected_pitcher
                    st.session_state.selected_hitters = selected_hitters
                    
                    # Filter movement data for charts and store in session state
                    movement_df = full_df[
                        (full_df["Batter"].isin(selected_hitters)) &
                        (full_df["MinDistToPitcher"] <= distance_threshold)
                    ].copy()
                    st.session_state.movement_df = movement_df
                    
                    st.success("✅ Analysis complete! Results displayed below.")
                else:
                    st.warning("❌ No sufficient data found for this matchup.")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display persistent results - OUTSIDE button block
    if 'summary_df' in st.session_state and 'breakdown_df' in st.session_state:
        st.markdown("---")
        st.header("📊 Results")
        
        # Main visualization
        fig = create_comprehensive_visualization(
            st.session_state.summary_df, 
            st.session_state.breakdown_df, 
            st.session_state.selected_pitcher
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Summary Statistics")
            st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🎯 Pitch Group Breakdown")
            st.dataframe(st.session_state.breakdown_df, use_container_width=True, hide_index=True)
        
        # Movement chart
        st.subheader("🌪️ Pitch Movement Chart")
        movement_fig = create_movement_chart(st.session_state.movement_df)
        st.plotly_chart(movement_fig, use_container_width=True)
    
    # Zone analysis - also outside button block
    if 'movement_df' in st.session_state and 'selected_hitters' in st.session_state:
        st.subheader("🎯 Zone-Level Heat Map Analysis")
        selected_hitter_heatmap = st.selectbox(
            "Select hitter for detailed zone analysis:",
            st.session_state.selected_hitters,
            key="heatmap_hitter"
        )
        
        if selected_hitter_heatmap:
            # This updates when dropdown changes
            hitter_data = st.session_state.movement_df[
                st.session_state.movement_df["Batter"] == selected_hitter_heatmap
            ].copy()
            
            if not hitter_data.empty:
                with st.spinner(f"Generating zone heatmap for {selected_hitter_heatmap}..."):
                    heatmap_img = generate_zone_heatmap(hitter_data, selected_hitter_heatmap)
                    st.markdown(f"<img src='{heatmap_img}' style='width: 100%; max-width: 1200px;'>", 
                              unsafe_allow_html=True)
            else:
                st.warning(f"No data available for {selected_hitter_heatmap} zone analysis.")
    
    # Coverage analysis and downloads - also outside button block
    if 'movement_df' in st.session_state and 'summary_df' in st.session_state:
        # Coverage analysis
        st.subheader("📈 Coverage Matrix")
        coverage_matrix = pd.DataFrame(
            index=st.session_state.selected_hitters, 
            columns=["Fastball", "Breaking", "Offspeed"]
        ).fillna(0)
        
        for hitter in st.session_state.selected_hitters:
            for group in ["Fastball", "Breaking", "Offspeed"]:
                matches = st.session_state.movement_df[
                    (st.session_state.movement_df["Batter"] == hitter) &
                    (st.session_state.movement_df["PitchGroup"] == group)
                ]
                coverage_matrix.loc[hitter, group] = len(matches)
        
        st.dataframe(coverage_matrix.astype(int), use_container_width=True)
        st.info("📊 Coverage Matrix shows pitch counts within distance threshold for each hitter vs pitch group combination")
        
        # Downloads
        st.subheader("💾 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_summary = st.session_state.summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv_summary,
                f"{st.session_state.selected_pitcher.replace(', ', '_')}_summary.csv",
                "text/csv"
            )
        
        with col2:
            csv_breakdown = st.session_state.breakdown_df.to_csv(index=False)
            st.download_button(
                "📥 Download Breakdown",
                csv_breakdown,
                f"{st.session_state.selected_pitcher.replace(', ', '_')}_breakdown.csv",
                "text/csv"
            )
        
        with col3:
            csv_movement = st.session_state.movement_df.to_csv(index=False)
            st.download_button(
                "📥 Download Pitch Data",
                csv_movement,
                f"{st.session_state.selected_pitcher.replace(', ', '_')}_pitch_level.csv",
                "text/csv"
            )
        
        # Analysis insights
        st.subheader("🔍 Analysis Insights")
        
        # Calculate insights
        best_matchup = st.session_state.summary_df.loc[st.session_state.summary_df["RV/100"].idxmin(), "Batter"] if not st.session_state.summary_df.empty else "N/A"
        worst_matchup = st.session_state.summary_df.loc[st.session_state.summary_df["RV/100"].idxmax(), "Batter"] if not st.session_state.summary_df.empty else "N/A"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Matchup (Pitcher)",
                best_matchup,
                f"RV/100: {st.session_state.summary_df['RV/100'].min():.2f}" if not st.session_state.summary_df.empty else "N/A"
            )
        
        with col2:
            st.metric(
                "Worst Matchup (Pitcher)",
                worst_matchup,
                f"RV/100: {st.session_state.summary_df['RV/100'].max():.2f}" if not st.session_state.summary_df.empty else "N/A"
            )
        
        with col3:
            avg_rv = st.session_state.summary_df["RV/100"].mean() if not st.session_state.summary_df.empty else 0
            st.metric(
                "Average RV/100",
                f"{avg_rv:.2f}",
                "Lower is better for pitcher"
            )

        # MOVE the Hot Arms analysis section to AFTER the main results display
    # Place this AFTER the "Analysis Insights" section at the very end:
    
    # Hot Arms Strategic Analysis
    if hot_arms and 'selected_hitters' in st.session_state:
        st.markdown("---")
        st.subheader("🔥 Hot Arms Strategic Analysis")
        
        if st.button("🎯 Analyze Hot Arms Strategy", type="secondary"):
            with st.spinner("Analyzing all pitcher-hitter matchups..."):
                matchups_df, pitcher_summaries = analyze_hot_arms_strategy(
                    hot_arms, st.session_state.selected_hitters, db_manager
                )
                
                if matchups_df is not None and not matchups_df.empty:
                    # Store in session state
                    st.session_state.hot_arms_matchups = matchups_df
                    st.session_state.hot_arms_summaries = pitcher_summaries
                    st.success("✅ Hot Arms analysis complete!")
                else:
                    st.warning("No data available for Hot Arms analysis")

    # Display Hot Arms results if available
    if 'hot_arms_matchups' in st.session_state and 'hot_arms_summaries' in st.session_state:
        
        # Pitcher Matchup Rankings with color coding
        st.subheader("🏆 Pitcher Matchup Rankings")
        st.write("**Color Guide:** 🟢 Great for Pitcher | 🟡 Neutral | 🔴 Bad for Pitcher")
        
        styled_rankings = create_matchup_rankings_table(
            st.session_state.hot_arms_matchups, 
            st.session_state.selected_hitters
        )
        if styled_rankings is not None:
            st.dataframe(styled_rankings, use_container_width=True)
        
        # Strategic Recommendations
        st.subheader("📍 Strategic Recommendations")
        recommendations = create_optimal_usage_recommendations(
            st.session_state.hot_arms_matchups, 
            st.session_state.hot_arms_summaries
        )
        
        # Display recommendations in organized columns
        if recommendations:
            for i in range(0, len(recommendations), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(recommendations):
                        rec = recommendations[i]
                        st.info(f"**{rec['type']}**\n\n{rec['recommendation']}\n\n*{rec['details']}*")
                
                with col2:
                    if i + 1 < len(recommendations):
                        rec = recommendations[i + 1]
                        st.info(f"**{rec['type']}**\n\n{rec['recommendation']}\n\n*{rec['details']}*")
        
        # Summary Statistics
        st.subheader("📊 Hot Arms Summary")
        summary_data = []
        for pitcher, stats in st.session_state.hot_arms_summaries.items():
            summary_data.append({
                'Pitcher': pitcher,
                'Avg RV/100': f"{stats['avg_rv']:.2f}",
                'Best Matchup': f"{stats['best_matchup']} ({stats['best_rv']:.2f})",
                'Worst Matchup': f"{stats['worst_matchup']} ({stats['worst_rv']:.2f})",
                'Total Pitches': stats['total_pitches']
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
