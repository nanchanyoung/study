import pandas as pd
from fbprophet import Prophet # (https://github.com/facebook/prophet)
import matplotlib.pyplot as plt

df = pd.read_csv(r'E:\python_study\avocado.csv')

# df.head()
# df.describe()

# type conventional 과 TotalUS 인 데이터만 뽑기
df = df.loc[(df.type == 'conventional') & (df.region == 'TotalUS')]

# string -> 날짜 형태로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 날짜, 평균가격만 담고 나머지는 다 지운다 (인덱스 초기화)
data = df[['Date', 'AveragePrice']].reset_index(drop=True)

#Prophet 사용을 위해 컬럼명 변경(https://facebook.github.io/prophet/docs/quick_start.html)
data = data.rename(columns={'Date': 'ds', 'AveragePrice': 'y'})

# string -> 날짜 형태로 변환
# data.plot(x='ds', y='y', figsize=(16, 8))

model = Prophet()
#데이터 학습
model.fit(data)

#예측할 데이터를 정의
future = model.make_future_dataframe(periods=365)

#앞으로 365일 예측 데이터를 담는다
forecast = model.predict(future)
# forecast.tail()

fig1 = model.plot((forecast))
fig2 = model.plot_components(forecast)
plt.show(fig1)
plt.show(fig2)