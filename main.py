"""
走行距離予測ダッシュボード
不規則な入力頻度の累積データから、将来の到達マイレージを予測するStreamlitアプリケーション。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import time
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# --- 定数設定 ---
DATA_FILE = "mileage_data.csv"
TARGET_YEARS = 5
TARGET_MILEAGE = 60000

def load_data() -> pd.DataFrame:
    """マイレージデータをCSVファイルから読み込み、正規化して返す。
    
    時系列予測モデルおよびバリデーションロジックが前提とする「時間的順序」を保証するため、
    不規則な入力順であっても常に日付の昇順でソートされた状態を提供する。
    
    Returns:
        pd.DataFrame: 'date'(datetime64)と'mileage'(int)を持つソート済みデータフレーム。
    """
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["date", "mileage"])
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df

def add_or_update_data(date: datetime.date | datetime.datetime, mileage: int, update: bool = False) -> None:
    """走行距離データをCSVへ永続化する。
    
    同一日付のデータは原則1件のみとするビジネスルールに基づき、
    updateフラグがTrueの場合は既存の同日データをパージしてから新しい値で再構成する。
    
    Args:
        date (datetime.date | datetime.datetime): 記録日
        mileage (int): 累積走行距離
        update (bool, optional): 既存の同日データを上書きするかどうか. Defaults to False.
    """
    df = load_data()
    date_pd = pd.to_datetime(date)
    
    if update:
        df = df[df["date"] != date_pd]
        
    new_row = pd.DataFrame([{"date": date_pd, "mileage": mileage}])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)

def check_validation(df_check: pd.DataFrame, target_date: datetime.date | datetime.datetime, target_mileage: int) -> tuple[bool, str]:
    """入力された走行距離が、既存の時系列データと物理的矛盾を起こさないか検証する。
    
    オドメーター（累積走行距離計）は時間経過に対して必ず単調増加するというドメイン知識に基づき、
    過去データより少ない値、または未来データより多い値の混入をブロックし、予測モデルの破綻を防ぐ。
    
    Args:
        df_check (pd.DataFrame): 比較対象となる既存データ
        target_date (datetime.date | datetime.datetime): 入力された記録日
        target_mileage (int): 入力された走行距離
        
    Returns:
        tuple[bool, str]: 検証結果(正常であればTrue)と、エラー時のユーザー向けメッセージ
    """
    target_date_pd = pd.to_datetime(target_date)
    future_invalid = df_check[(df_check['date'] > target_date_pd) & (df_check['mileage'] < target_mileage)]
    past_invalid = df_check[(df_check['date'] < target_date_pd) & (df_check['mileage'] > target_mileage)]

    if not future_invalid.empty:
        conflict_date = future_invalid.iloc[0]['date'].strftime('%Y/%m/%d')
        conflict_dist = future_invalid.iloc[0]['mileage']
        return False, f"入力エラー: 未来の {conflict_date} に {conflict_dist}km と記録されています。これより大きい距離を過去の日付には登録できません。"
    elif not past_invalid.empty:
        conflict_date = past_invalid.iloc[-1]['date'].strftime('%Y/%m/%d')
        conflict_dist = past_invalid.iloc[-1]['mileage']
        return False, f"入力エラー: 過去の {conflict_date} に {conflict_dist}km と記録されています。これより小さい距離を未来の日付には登録できません。"
    
    return True, ""

# --- UI構築 ---
st.set_page_config(page_title="走行距離予測ダッシュボード", layout="wide")

# --- 簡易パスワード認証 ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 認証が必要です")
    pw = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if pw == st.secrets["app_password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    st.stop()

st.title("🚗 走行距離 予測＆管理ダッシュボード")

# Streamlitの再描画アーキテクチャにおいて、確認ダイアログ（上書きのYes/No）の
# コンテキストをボタン押下後も維持するためにsession_stateを利用する。
if 'confirm_overwrite' not in st.session_state:
    st.session_state.confirm_overwrite = False
if 'pending_data' not in st.session_state:
    st.session_state.pending_data = None

df_current = load_data()

with st.container():
    st.markdown("### 📝 記録を入力")
    
    if st.session_state.confirm_overwrite:
        p_data = st.session_state.pending_data
        st.warning(f"⚠️ **{p_data['date'].strftime('%Y/%m/%d')}** には既に **{p_data['old_mileage']} km** の記録があります。\n\n**{p_data['mileage']} km** に上書きしてよろしいですか？")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("はい、上書きします", use_container_width=True):
                # 自身を上書きする際、古い同日データが単調増加のバリデーションに
                # 誤検知されるのを防ぐため、一時的にチェック対象から除外する。
                df_temp = df_current[df_current['date'] != pd.to_datetime(p_data['date'])]
                is_valid, error_msg = check_validation(df_temp, p_data['date'], p_data['mileage'])
                if is_valid:
                    add_or_update_data(p_data['date'], p_data['mileage'], update=True)
                    st.toast("✅ データを上書き保存しました！") 
                    time.sleep(1.5) 
                else:
                    st.error(f"🚨 {error_msg}")
                    time.sleep(2) 
                
                st.session_state.confirm_overwrite = False
                st.session_state.pending_data = None
                st.rerun()
        with col_no:
            if st.button("キャンセル", use_container_width=True):
                st.session_state.confirm_overwrite = False
                st.session_state.pending_data = None
                st.rerun()
                
    else:
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_date = st.date_input("記録日", datetime.datetime.today())
            with col2:
                input_mileage = st.number_input("累積走行距離 (km)", min_value=0, step=100)
            submitted = st.form_submit_button("記録を保存")
            
            if submitted:
                input_date_pd = pd.to_datetime(input_date)
                existing_record = df_current[df_current['date'] == input_date_pd]
                
                if not existing_record.empty:
                    old_mileage = existing_record.iloc[0]['mileage']
                    if old_mileage == input_mileage:
                        st.info(f"✅ {input_date.strftime('%Y/%m/%d')} は既に {input_mileage} km で登録されています。")
                    else:
                        st.session_state.confirm_overwrite = True
                        st.session_state.pending_data = {
                            'date': input_date, 
                            'mileage': input_mileage, 
                            'old_mileage': old_mileage
                        }
                        st.rerun()
                else:
                    is_valid, error_msg = check_validation(df_current, input_date, input_mileage)
                    if is_valid:
                        add_or_update_data(input_date, input_mileage, update=False)
                        st.toast("✅ データを保存しました！") 
                        time.sleep(1.5) 
                        st.rerun()
                    else:
                        st.error(f"🚨 {error_msg}")

st.divider()
df = load_data()

# 予測モデル（特に回帰分析）は自由度が最低限必要であり、
# 2件以下のデータでは過学習や計算不能に陥るためフェイルセーフとして早期リターンさせる。
if len(df) < 3:
    st.info("📊 予測モデルを構築して将来をシミュレーションするには、最低3回分のデータが必要です。記録を続けてください！")
    if not df.empty:
        st.dataframe(df)
else:
    # 記録タイミングの不規則性を吸収するため、絶対日時ではなく
    # 「観測開始からの経過日数」という連続的なスカラー値に変換して線形回帰に適合させる。
    df['days_passed'] = (df['date'] - df['date'].min()).dt.days
    
    zero_km_dates = df[df['mileage'] == 0]['date']
    if not zero_km_dates.empty:
        start_date = zero_km_dates.min()
    else:
        start_date = df['date'].min()
        
    try:
        target_date = start_date.replace(year=start_date.year + TARGET_YEARS)
    except ValueError:
        # 起点日がうるう年の2月29日の場合、5年後には同日が存在せず例外が発生するため、
        # 365日 * 5年 + 1日として安全にターゲット日を算出する。
        target_date = start_date + timedelta(days=365 * TARGET_YEARS + 1)
        
    target_days_passed = (target_date - df['date'].min()).days
    target_X_df = pd.DataFrame({'days_passed': [target_days_passed]})

    # --- モデル1: 全期間の線形回帰 ---
    X_all = df[['days_passed']]
    y_all = df['mileage']
    model_all = LinearRegression().fit(X_all, y_all)
    pred_target_all = model_all.predict(target_X_df)[0]
    
    # --- モデル2: 直近3ヶ月の線形回帰 ---
    cutoff_date = df['date'].max() - timedelta(days=90)
    df_recent = df[df['date'] >= cutoff_date]
    if len(df_recent) < 3:
        df_recent = df.tail(3)
        
    X_recent = df_recent[['days_passed']]
    y_recent = df_recent['mileage']
    model_recent = LinearRegression().fit(X_recent, y_recent)
    pred_target_recent = model_recent.predict(target_X_df)[0]

    # --- モデル3: 状態空間モデル (Local Linear Trend) ---
    # 【仕組みの概要】
    # 状態空間モデルは、「観測される値（ユーザーが入力した距離）」の背後に、「観測できない真の状態（実際の正確な距離と、現在の走行ペース）」が
    # 存在すると仮定して推論を行うアプローチです。カルマンフィルタというアルゴリズムを用いて、観測ノイズ（入力のブレ）と
    # システムノイズ（ライフスタイルの変化によるペースの変動）を分離しながら、現在の「真のペース」を推定します。
    
    # statsmodelsライブラリはカレンダー通りに連続した等間隔の時系列データを前提とするため、
    # まずは不規則な入力データを日次データ(入力がない日はNaN)としてカレンダー状に再構築します。
    # カルマンフィルタは、NaNの日を自動的に予測で補完（フィルタリング）しながら状態を滑らかに更新していきます。
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df_ssm = pd.DataFrame({'date': date_range})
    df_ssm = df_ssm.merge(df[['date', 'mileage']], on='date', how='left')
    df_ssm.set_index('date', inplace=True)
    
    # 【モデル設定: Local Linear Trend】
    # 'local linear trend'（ローカル線形トレンド）という設定は、レベル(現在の累積距離)とトレンド(現在のペース)の
    # 両方が時間とともに確率的に変動（ランダムウォーク）すると仮定するモデルです。
    # これにより、単なる直線ではなく「最近ちょっと車に乗らなくなった」といったペースの変化に追従する滑らかな予測が可能になります。
    model_ssm = sm.tsa.UnobservedComponents(df_ssm['mileage'], level='local linear trend')
    res_ssm = model_ssm.fit(disp=False)
    
    target_date_pd = pd.to_datetime(target_date)
    steps_to_target = (target_date_pd - df['date'].max()).days
    forecast_steps = max(365 * 5, steps_to_target + 30)
    
    forecast_ssm = res_ssm.get_forecast(steps=forecast_steps)
    
    if target_date_pd in forecast_ssm.predicted_mean.index:
        pred_target_ssm = forecast_ssm.predicted_mean[target_date_pd]
    else:
        # インデックス外へのアクセスを防ぐためのフォールバック
        pred_target_ssm = forecast_ssm.predicted_mean.iloc[-1]
    
    st.subheader(f"🎯 5年後の到達予測診断 (目標: {TARGET_MILEAGE:,.0f} km以内)")
    st.caption(f"※ 起点日 ({start_date.strftime('%Y/%m/%d')}) から5年後の **{target_date.strftime('%Y/%m/%d')}** 時点の累積距離を予測・評価します。")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**📊 1. 全期間トレンド**")
        if pred_target_all > TARGET_MILEAGE:
            st.warning(f"⚠️ **注意**\n\n予測: **{pred_target_all:,.0f}** km\n\n長期的ペースでは5年後に目標をオーバーする予測です。")
        else:
            st.success(f"✅ **順調**\n\n予測: **{pred_target_all:,.0f}** km\n\n長期的ペースでは5年後も目標内に収まります。")

    with col_b:
        st.markdown("**📈 2. 直近3ヶ月トレンド**")
        if pred_target_recent > TARGET_MILEAGE:
            st.error(f"🚨 **警告**\n\n予測: **{pred_target_recent:,.0f}** km\n\n最近のペースが続くと5年後に目標を激しくオーバーします！")
        else:
            st.success(f"✅ **順調**\n\n予測: **{pred_target_recent:,.0f}** km\n\n最近のペースでも5年後に目標内に収まります。")

    with col_c:
        st.markdown("**🤖 3. 状態空間モデル予測**")
        if pred_target_ssm > TARGET_MILEAGE:
            st.warning(f"⚠️ **注意**\n\n予測: **{pred_target_ssm:,.0f}** km\n\nカルマンフィルタの予測でも5年後に目標をオーバーします。")
        else:
            st.success(f"✅ **順調**\n\n予測: **{pred_target_ssm:,.0f}** km\n\nカルマンフィルタの予測でも5年後に目標内に収まります。")

    target_days_from_last = (target_date - df['date'].max()).days
    future_days = max(365 * 5, target_days_from_last + 30) 
    
    last_day = df['days_passed'].max()
    pred_X_array = np.arange(0, last_day + future_days, 30) 
    
    # scikit-learnモデルのpredict実行時における feature_names の不一致警告を防ぐため、
    # 生のnumpy配列ではなく、学習時と同じ特徴量名を持つDataFrameを明示的に渡す。
    pred_X_df = pd.DataFrame({'days_passed': pred_X_array})
    pred_dates = [df['date'].min() + timedelta(days=int(d)) for d in pred_X_array]
    
    # --- 1. 全期間モデルの信頼区間計算 ---
    pred_y_all = model_all.predict(pred_X_df)
    residuals_all = y_all - model_all.predict(X_all)
    std_all = np.std(residuals_all)
    
    # 遠い未来ほど不確実性が高まる性質を表現するため、時間経過に比例してブレ幅をスケールさせる。
    scale_all = 1 + np.maximum(0, (pred_X_array - last_day) / (365 * 2))
    margin_all = std_all * scale_all * 1.96
    upper_all = pred_y_all + margin_all
    lower_all = pred_y_all - margin_all

    # --- 2. 直近モデルの信頼区間計算 ---
    pred_y_recent = model_recent.predict(pred_X_df)
    residuals_recent = y_recent - model_recent.predict(X_recent)
    std_recent = np.std(residuals_recent)
    
    # 直近のサンプルサイズが極端に少なく分散がゼロに近い場合、信頼区間が潰れて視認性が悪化するため、
    # 全期間の分散に依存した最低限の不確実性の幅（ブレ）を担保する。
    if std_recent < 10:
        std_recent = max(50, np.std(y_all) * 0.5)
        
    scale_recent = 1 + np.maximum(0, (pred_X_array - last_day) / 365)
    margin_recent = std_recent * scale_recent * 1.96
    upper_recent = pred_y_recent + margin_recent
    lower_recent = pred_y_recent - margin_recent
    
    # --- 3. 状態空間モデルの信頼区間（予測範囲）取得 ---
    # 【予測範囲（ブレ幅）の性質】
    # 線形回帰のブレ幅が疑似的にスケーリングして計算されているのに対し、状態空間モデルの予測区間は
    # 「ペースが今後どう変わるかわからない」という確率的な不確実性が、時間経過とともに積分されて蓄積していく
    # 性質（確率過程）を数学的に正確に表現しています。そのため、遠い未来ほど自然にラッパ状に広がっていきます。
    # ここでは alpha=0.05 を指定し、95%の確率で将来の距離が収まる範囲（95%予測区間）を取得しています。
    pred_dates_ssm = forecast_ssm.predicted_mean.index.to_pydatetime()
    pred_y_ssm = forecast_ssm.predicted_mean.values
    ssm_ci = forecast_ssm.conf_int(alpha=0.05)
    ssm_lower = ssm_ci.iloc[:, 0].values
    ssm_upper = ssm_ci.iloc[:, 1].values
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚙️ グラフ表示設定")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_m1 = st.checkbox("1. 全期間予測を表示", value=True)
        show_ci1 = st.checkbox("1. 信頼区間を表示", value=False, key="ci1")
    with col_opt2:
        show_m2 = st.checkbox("2. 直近予測を表示", value=True)
        show_ci2 = st.checkbox("2. 信頼区間を表示", value=False, key="ci2")
    with col_opt3:
        show_m3 = st.checkbox("3. 状態空間モデル予測を表示", value=True)
        show_ci3 = st.checkbox("3. 信頼区間を表示", value=True, key="ci3")

    fig = go.Figure()
    
    if show_ci1:
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1], y=list(upper_all) + list(lower_all)[::-1],
            fill='toself', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='全期間ブレ幅'
        ))
    if show_m1:
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_y_all, mode='lines', 
                                 name=f'全期間トレンド予測 (5年後予想: {pred_target_all:,.0f}km)', 
                                 line=dict(color='gray', dash='dash')))

    if show_ci2:
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1], y=list(upper_recent) + list(lower_recent)[::-1],
            fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='直近ブレ幅'
        ))
    if show_m2:
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_y_recent, mode='lines', 
                                 name=f'直近3ヶ月トレンド予測 (5年後予想: {pred_target_recent:,.0f}km)', 
                                 line=dict(color='orange', dash='dot', width=3)))

    if show_ci3:
        fig.add_trace(go.Scatter(
            x=list(pred_dates_ssm) + list(pred_dates_ssm)[::-1], y=list(ssm_upper) + list(ssm_lower)[::-1],
            fill='toself', fillcolor='rgba(0, 128, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='SSMブレ幅'
        ))
    if show_m3:
        fig.add_trace(go.Scatter(x=pred_dates_ssm, y=pred_y_ssm, mode='lines', 
                                 name=f'状態空間モデル予測 (5年後予想: {pred_target_ssm:,.0f}km)', 
                                 line=dict(color='green', dash='solid', width=2)))
                             
    fig.add_trace(go.Scatter(x=df['date'], y=df['mileage'], mode='lines+markers', 
                             name='実績データ', line=dict(color='blue', width=2), marker=dict(size=8)))
                             
    fig.add_hline(y=TARGET_MILEAGE, line_dash="dot", line_color="red", 
                  annotation_text=f"目標上限 ({TARGET_MILEAGE:,.0f}km)", annotation_position="top left")
    
    # Plotlyの内部仕様として、日時型(datetime)をX座標に持つ垂直線に対してテキストアノテーションを
    # 併記すると計算エラー(TypeError)が起きるため、描画処理とテキスト処理を分けて実装している。
    fig.add_vline(x=target_date, line_dash="dot", line_color="purple")
    fig.add_annotation(
        x=target_date, y=0.95, yref="paper", 
        text="評価日 (起点から5年後)", showarrow=False, xanchor="left", xshift=5, font=dict(color="purple", size=12)
    )
    
    fig.update_layout(
        title="累積走行距離の推移と将来予測",
        xaxis_title="日付",
        yaxis_title="累積走行距離 (km)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📂 登録済みの生データを確認"):
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)