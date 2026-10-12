"""Train demo CNN/LSTM models on SYNTHETIC data. Not valid for real-world crime forecasting."""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; MODELS=ROOT/'models'
MODELS.mkdir(exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)

DISTRICTS=['Thiruvananthapuram','Kollam','Pathanamthitta','Alappuzha','Kottayam','Idukki','Ernakulam','Thrissur','Palakkad','Malappuram','Kozhikode','Wayanad','Kannur','Kasaragod']
AREAS={
'Thiruvananthapuram':['Thiruvananthapuram City','Neyyattinkara','Attingal','Varkala'], 'Kollam':['Kollam City','Karunagappally','Punalur','Kottarakkara'],
'Pathanamthitta':['Pathanamthitta','Adoor','Thiruvalla','Ranni'],'Alappuzha':['Alappuzha','Cherthala','Kayamkulam','Mavelikkara'],
'Kottayam':['Kottayam','Changanassery','Pala','Vaikom'],'Idukki':['Thodupuzha','Munnar','Kattappana','Adimali'],
'Ernakulam':['Kochi','Aluva','Perumbavoor','Kothamangalam'],'Thrissur':['Thrissur','Pazhayannur','Chelakkara','Lakkidi','Chalakudy'],
'Palakkad':['Palakkad','Ottapalam','Mannarkkad','Pattambi'],'Malappuram':['Malappuram','Manjeri','Tirur','Perinthalmanna'],
'Kozhikode':['Kozhikode','Vadakara','Koyilandy','Thamarassery'],'Wayanad':['Kalpetta','Sulthan Bathery','Mananthavady','Vythiri'],
'Kannur':['Kannur','Thalassery','Payyannur','Iritty'],'Kasaragod':['Kasaragod','Kanhangad','Nileshwar','Uppala']}
# Synthetic rows intentionally generated for software demonstration, never presented as actual incidents.
N=60000
rows=[]
for i in range(N):
    district=random.choice(DISTRICTS); area=random.choice(AREAS[district]); month=random.randint(1,12); year=random.randint(2018,2025)
    hour=random.randrange(24); weekend=random.randint(0,1); holiday=random.binomial(1,.08) if hasattr(random,'binomial') else int(random.random()<.08)
    lighting=float(np.clip(np.random.normal(0.65 if 6<=hour<=18 else 0.35,0.18),0,1))
    footfall=float(np.clip(np.random.lognormal(2.1 + (0.45 if district in ['Ernakulam','Thiruvananthapuram','Kozhikode'] else 0),.55),0,100))
    transit=float(np.clip(np.random.beta(2,3)+(0.12 if area in ['Kochi','Thrissur','Thiruvananthapuram City'] else 0),0,1))
    night=int(hour>=20 or hour<5); late_night=int(hour>=23 or hour<4)
    seasonal=0.08*np.sin((month/12)*2*np.pi)
    district_bias={'Ernakulam':.12,'Thiruvananthapuram':.07,'Kozhikode':.05,'Thrissur':.03}.get(district,0)
    # Target is a simulated label generated from assumptions, not observed crime.
    score=-1.4 + .8*night + .65*late_night + .35*weekend + .3*holiday + .6*(1-lighting) + .012*footfall + .35*transit + district_bias + seasonal + np.random.normal(0,.35)
    risk=int(score>0.55)
    rows.append((year,month,hour,weekend,holiday,lighting,footfall,transit,night,late_night,DISTRICTS.index(district),area,district,score,risk))
df=pd.DataFrame(rows,columns=['year','month','hour','weekend','holiday','lighting_index','footfall_index','transit_index','night','late_night','district_id','area','district','synthetic_risk_score','synthetic_risk_label'])
df.to_csv(DATA/'synthetic_demo_events.csv',index=False)
features=['year','month','hour','weekend','holiday','lighting_index','footfall_index','transit_index','night','late_night','district_id']
X=df[features].astype(float).values; y=df['synthetic_risk_label'].values.astype(np.float32)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
scaler=StandardScaler().fit(Xtr); Xtr=scaler.transform(Xtr).astype('float32'); Xte=scaler.transform(Xte).astype('float32')
with open(MODELS/'cnn_scaler.json','w') as f: json.dump({'features':features,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},f,indent=2)
class CrimeCNN(nn.Module):
 def __init__(self,n):
  super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(32,16),nn.ReLU(),nn.Dropout(.15),nn.Linear(16,1))
 def forward(self,x): return self.net(x.unsqueeze(1)).squeeze(1)
cnn=CrimeCNN(len(features)); opt=torch.optim.Adam(cnn.parameters(),lr=.001); lossfn=nn.BCEWithLogitsLoss()
train_loader=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=512,shuffle=True)
for epoch in range(8):
 cnn.train()
 for xb,yb in train_loader:
  opt.zero_grad(); loss=lossfn(cnn(xb),yb); loss.backward(); opt.step()
cnn.eval()
with torch.no_grad(): probs=torch.sigmoid(cnn(torch.tensor(Xte))).numpy(); pred=(probs>=.5).astype(int)
cnn_acc=float(accuracy_score(yte,pred)); cnn_f1=float(f1_score(yte,pred,zero_division=0))
torch.save({'state_dict':cnn.state_dict(),'features':features,'model_type':'synthetic_demo_only'},MODELS/'cnn_demo_weights.pt')

# Build synthetic monthly sequences per district-area; target is next month synthetic risk mean.
monthly=[]
for district in DISTRICTS:
 for area in AREAS[district]:
  base=np.random.uniform(.15,.72)
  for t,period in enumerate(pd.period_range('2018-01','2025-12',freq='M')):
   seasonal=.08*np.sin((period.month/12)*2*np.pi)
   # simulated sequence, not actual counts or risk observations
   level=float(np.clip(base + seasonal + .07*np.sin(t/7)+np.random.normal(0,.07),0,1))
   monthly.append({'district':district,'area':area,'date':str(period),'month_index':t,'synthetic_monthly_risk':level,
                   'night_share':float(np.clip(.3+level*.25+np.random.normal(0,.04),0,1)),
                   'weekend_share':float(np.clip(.28+np.random.normal(0,.03),0,1)),
                   'lighting_index':float(np.clip(.7-level*.3+np.random.normal(0,.04),0,1)),
                   'footfall_index':float(np.clip(.3+level*.5+np.random.normal(0,.08),0,1)),
                   'synthetic_event_count':int(max(0,np.random.poisson(10+level*18)))})
md=pd.DataFrame(monthly); md.to_csv(DATA/'synthetic_demo_monthly_series.csv',index=False)
seq_feats=['synthetic_monthly_risk','night_share','weekend_share','lighting_index','footfall_index']
seqs=[]; targets=[]; groups=[]
for (district,area),g in md.groupby(['district','area']):
 g=g.sort_values('month_index'); vals=g[seq_feats].values.astype('float32')
 for i in range(len(vals)-13):
  seqs.append(vals[i:i+12]); targets.append(vals[i+12,0]); groups.append((district,area))
seqs=np.array(seqs); targets=np.array(targets,dtype='float32')
idx=np.arange(len(seqs)); np.random.shuffle(idx); cut=int(.8*len(idx)); itr,ite=idx[:cut],idx[cut:]
# per feature scale from all sequences; only synthetic data
mean=seqs[itr].reshape(-1,len(seq_feats)).mean(axis=0); std=seqs[itr].reshape(-1,len(seq_feats)).std(axis=0)+1e-6
seqs=(seqs-mean)/std
with open(MODELS/'lstm_scaler.json','w') as f: json.dump({'features':seq_feats,'mean':mean.tolist(),'scale':std.tolist(),'sequence_length':12},f,indent=2)
class CrimeLSTM(nn.Module):
 def __init__(self,nf):
  super().__init__(); self.lstm=nn.LSTM(nf,32,num_layers=1,batch_first=True); self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
 def forward(self,x): out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(1)
lstm=CrimeLSTM(len(seq_feats)); opt=torch.optim.Adam(lstm.parameters(),lr=.001); mse=nn.MSELoss()
loader=DataLoader(TensorDataset(torch.tensor(seqs[itr]),torch.tensor(targets[itr])),batch_size=256,shuffle=True)
for epoch in range(12):
 lstm.train()
 for xb,yb in loader:
  opt.zero_grad(); loss=mse(lstm(xb),yb); loss.backward(); opt.step()
lstm.eval()
with torch.no_grad(): lp=lstm(torch.tensor(seqs[ite])).numpy(); mae=float(mean_absolute_error(targets[ite],lp))
torch.save({'state_dict':lstm.state_dict(),'features':seq_feats,'sequence_length':12,'model_type':'synthetic_demo_only'},MODELS/'lstm_demo_weights.pt')
metrics={'warning':'All metrics are measured against synthetic labels generated by assumptions; they are NOT real-world crime prediction accuracy.', 'dataset_rows':int(len(df)), 'monthly_sequences':int(len(seqs)), 'cnn_synthetic_holdout_accuracy':cnn_acc,'cnn_synthetic_holdout_f1':cnn_f1,'lstm_synthetic_holdout_mae':mae,'seed':SEED}
(MODELS/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
