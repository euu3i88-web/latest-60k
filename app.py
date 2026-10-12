from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODELS=ROOT/'models'
DISTRICT_AREAS={
'Thiruvananthapuram':['Thiruvananthapuram City','Neyyattinkara','Attingal','Varkala'], 'Kollam':['Kollam City','Karunagappally','Punalur','Kottarakkara'],
'Pathanamthitta':['Pathanamthitta','Adoor','Thiruvalla','Ranni'],'Alappuzha':['Alappuzha','Cherthala','Kayamkulam','Mavelikkara'],
'Kottayam':['Kottayam','Changanassery','Pala','Vaikom'],'Idukki':['Thodupuzha','Munnar','Kattappana','Adimali'],
'Ernakulam':['Kochi','Aluva','Perumbavoor','Kothamangalam'],'Thrissur':['Thrissur','Pazhayannur','Chelakkara','Lakkidi','Chalakudy'],
'Palakkad':['Palakkad','Ottapalam','Mannarkkad','Pattambi'],'Malappuram':['Malappuram','Manjeri','Tirur','Perinthalmanna'],
'Kozhikode':['Kozhikode','Vadakara','Koyilandy','Thamarassery'],'Wayanad':['Kalpetta','Sulthan Bathery','Mananthavady','Vythiri'],
'Kannur':['Kannur','Thalassery','Payyannur','Iritty'],'Kasaragod':['Kasaragod','Kanhangad','Nileshwar','Uppala']}

class CrimeCNN(nn.Module):
    def __init__(self,n):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(32,16),nn.ReLU(),nn.Dropout(.15),nn.Linear(16,1))
    def forward(self,x): return self.net(x.unsqueeze(1)).squeeze(1)
class CrimeLSTM(nn.Module):
    def __init__(self,nf):
        super().__init__(); self.lstm=nn.LSTM(nf,32,num_layers=1,batch_first=True); self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,x): out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(1)

@st.cache_resource
def load_models():
    csc=json.loads((MODELS/'cnn_scaler.json').read_text()); lsc=json.loads((MODELS/'lstm_scaler.json').read_text())
    cm=CrimeCNN(len(csc['features'])); cp=torch.load(MODELS/'cnn_demo_weights.pt',map_location='cpu',weights_only=False); cm.load_state_dict(cp['state_dict']); cm.eval()
    lm=CrimeLSTM(len(lsc['features'])); lp=torch.load(MODELS/'lstm_demo_weights.pt',map_location='cpu',weights_only=False); lm.load_state_dict(lp['state_dict']); lm.eval()
    return cm,csc,lm,lsc

def read_csv(path):
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

st.set_page_config(page_title='CrimeLens Kerala | Research Demo',page_icon='🧭',layout='wide')
st.title('🧭 CrimeLens Kerala')
st.caption('Kerala crime statistics dashboard + experimental CNN/LSTM demonstration')
st.warning('Research limitation: the included CNN/LSTM weights were trained on synthetic demonstration data, not verified local crime incidents. Model scores below are simulated model outputs, NOT validated real-world crime forecasts and must not be used to decide whether a place is safe or unsafe.')

with st.sidebar:
    st.header('Explore location')
    district=st.selectbox('District',list(DISTRICT_AREAS.keys()),index=7)
    area=st.selectbox('Town / area',DISTRICT_AREAS[district],index=1 if district=='Thrissur' else 0)
    st.markdown('---')
    st.caption('Official records are shown separately from synthetic model demos.')

stats=read_csv(DATA/'kerala_police_statewide_ipc_bns_2020_2026.csv')
station=read_csv(DATA/'pazhayannur_police_station_stats.csv')
monthly=read_csv(DATA/'synthetic_demo_monthly_series.csv')
metrics=json.loads((MODELS/'training_metrics.json').read_text())

st.subheader('Official historical statistics')
if not stats.empty:
    heads=sorted(stats['crime_head'].dropna().unique())
    chosen_head=st.selectbox('Crime category (official statewide counts)',heads,index=heads.index('Theft') if 'Theft' in heads else 0)
    sub=stats[stats.crime_head==chosen_head].copy().sort_values('year')
    if not sub.empty:
        chart=sub.set_index('year')[['registered_cases']]
        st.line_chart(chart)
        st.caption('Source: Kerala Police official statistics. These figures are statewide totals, not town-level estimates. The 2026 figures are partial-year where noted.')
        st.dataframe(sub[['year','registered_cases','period_note','source_name','source_url']],use_container_width=True,hide_index=True)
    st.markdown('[Kerala Police — IPC/BNS statistics](https://keralapolice.gov.in/crime-statistics/ipc-cases) · [Kerala Police — total cases](https://keralapolice.gov.in/crime/total-cases) · [NCRB / OGD crime data catalog](https://kerala.data.gov.in/catalog/crime-india-2021)')
if district=='Thrissur' and area in ['Pazhayannur','Chelakkara','Lakkidi'] and not station.empty:
    st.subheader('Pazhayannur Police Station published history')
    s=station[station['crime_head'].astype(str).str.contains('Total cases',case=False,na=False)].sort_values('year')
    if not s.empty:
        st.bar_chart(s.set_index('year')[['registered_cases']])
        st.caption('Official station-level figures available in this bundled extract cover historical annual totals only. They are not a live or complete crime feed for Pazhayannur, Chelakkara, or Lakkidi.')

st.divider(); st.subheader('CNN demonstration: user-selected scenario')
st.write('Try different time and environmental inputs. This shows how the included CNN runs end-to-end; it is not a calibrated real-world safety assessment.')
col1,col2,col3=st.columns(3)
with col1: year=st.selectbox('Year',list(range(2026,2017,-1)),index=0); month=st.slider('Month',1,12,10); hour=st.slider('Hour of day',0,23,20)
with col2: weekend=st.selectbox('Weekend?', ['No','Yes']); holiday=st.selectbox('Holiday?', ['No','Yes']); lighting=st.slider('Lighting index (demo input)',0.0,1.0,0.5,.05)
with col3: footfall=st.slider('Footfall index (demo input)',0.0,100.0,20.0,1.0); transit=st.slider('Transit access index',0.0,1.0,.4,.05)
if st.button('Run CNN demo',type='primary'):
    try:
        cnn,csc,_,_=load_models()
        vals={'year':year,'month':month,'hour':hour,'weekend':int(weekend=='Yes'),'holiday':int(holiday=='Yes'),'lighting_index':lighting,'footfall_index':footfall,'transit_index':transit,'night':int(hour>=20 or hour<5),'late_night':int(hour>=23 or hour<4),'district_id':list(DISTRICT_AREAS).index(district)}
        arr=np.array([[vals[k] for k in csc['features']]],dtype=np.float32)
        # JSON scaler values default to NumPy float64; force float32 after scaling to match PyTorch weights.
        arr=((arr-np.asarray(csc['mean'],dtype=np.float32))/np.asarray(csc['scale'],dtype=np.float32)).astype(np.float32,copy=False)
        x=torch.as_tensor(arr,dtype=torch.float32)
        cnn=cnn.float()
        with torch.no_grad(): prob=float(torch.sigmoid(cnn(x)).item())
        st.metric('Synthetic-model score',f'{prob:.1%}')
        st.write('Demo class:', '**higher simulated label**' if prob>=.5 else '**lower simulated label**')
        st.caption('This score estimates the synthetic label pattern used during training. It is not a probability of a crime occurring.')
    except Exception as e: st.error(f'Model could not run: {e}')

st.divider(); st.subheader('LSTM demonstration: synthetic monthly sequence')
if not monthly.empty:
    m=monthly[(monthly.district==district)&(monthly.area==area)].sort_values('month_index')
    if len(m)>=13:
        st.line_chart(m.set_index('date')[['synthetic_monthly_risk']].tail(36))
        st.caption('Every value in this chart is generated synthetic demonstration data, not observed local crime history.')
        if st.button('Run LSTM next-month demo'):
            try:
                _,_,lstm,lsc=load_models(); feats=lsc['features']; tail=m[feats].tail(12).to_numpy(dtype=np.float32)
                # Keep normalized sequence float32: NumPy float64 scaler arrays otherwise promote the result.
                tail=((tail-np.asarray(lsc['mean'],dtype=np.float32))/np.asarray(lsc['scale'],dtype=np.float32)).astype(np.float32,copy=False)
                x=torch.as_tensor(tail[None,:,:],dtype=torch.float32)
                lstm=lstm.float()
                with torch.no_grad(): prediction=float(lstm(x).item())
                st.metric('Next-month synthetic index',f'{prediction:.3f}')
                st.caption('Forecast of a synthetic index only; not a forecast of actual crimes.')
            except Exception as e: st.error(f'LSTM could not run: {e}')

st.divider(); st.subheader('Model training report')
st.write(f"CNN synthetic holdout accuracy: **{metrics['cnn_synthetic_holdout_accuracy']:.1%}** · Synthetic F1: **{metrics['cnn_synthetic_holdout_f1']:.3f}** · LSTM synthetic MAE: **{metrics['lstm_synthetic_holdout_mae']:.3f}**")
st.caption('These metrics measure agreement with generated labels only. They do not measure real-world accuracy. See README for training instructions and limitations.')
with st.expander('What is included?'):
    st.markdown('- Official statewide annual crime-head counts, transcribed from Kerala Police public statistics.\n- Historical Pazhayannur Police Station total-case extract.\n- 60,000 generated synthetic rows and synthetic monthly sequences.\n- Saved CNN and LSTM weights plus scalers.\n- Reproducible training script.\n- All 14 Kerala districts and sample town/area search options, including Pazhayannur, Chelakkara, and Lakkidi.')
