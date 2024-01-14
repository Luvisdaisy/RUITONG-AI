import pickle

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 数据字典
asp = {"标准": "std", "涡轮": "turbo"}
drivew = {"后驱": "rwd", "前驱": "fwd", "四驱": "4wd"}
cylnum = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    8: "eight",
    12: "twelve",
}
dn = {"双门": "two", "四门": "four"}

el = {"前置": "front", "后置": "rear"}
# 原始df字段名
df = pd.read_csv("../data/original_df.csv")
df.drop("price", axis=1, inplace=True)
cols = df.columns
# 独热向量编码过后的字段名
dummy = pd.read_csv("../data/dummy_df.csv")
cols_to_use = dummy.columns
# 车品牌名
cars = df["CarName"].unique().tolist()
carNameCap = []
for col in cars:
    carNameCap.append(col.capitalize())
# fuel
fuel = df["fueltype"].unique().tolist()
fuelCap = []
for fu in fuel:
    fuelCap.append(fu.capitalize())
# For carbon, engine type, fuel system
carb = df["carbody"].unique().tolist()
engtype = df["enginetype"].unique().tolist()
fuelsys = df["fuelsystem"].unique().tolist()


def transform(data):
    # 数值型幅度缩放
    sc = StandardScaler()
    # 导入模型
    lasso_reg = pickle.load(open("../model.pkl", "rb"))
    # 新数据Dataframe
    new_df = pd.DataFrame([data], columns=cols)
    # 切分类别型与数值型字段
    cat = []
    num = []
    for col in new_df.columns:
        if new_df[col].dtypes == "object":
            cat.append(col)
        else:
            num.append(col)
    # 构建模型所需数据格式
    x1_new = pd.get_dummies(new_df[cat], drop_first=False)
    x2_new = new_df[num]
    X_new = pd.concat([x2_new, x1_new], axis=1)
    final_df = pd.DataFrame(columns=cols_to_use)
    final_df = pd.concat([final_df, X_new])
    final_df = final_df.fillna(0)
    final_df = pd.concat([final_df, dummy])
    X_new = final_df.values
    X_new[:, : (len(x1_new.columns))] = sc.fit_transform(
        X_new[:, : (len(x1_new.columns))]
    )
    print(X_new[-1].reshape(-1, 1))
    output = lasso_reg.predict(X_new[-1].reshape(1, -1))
    return "The price of the car " + str(round(np.exp(output)[0], 2)) + "$"


# 预估价格的主函数
def predict_price(
    car,
    fueltype,
    aspiration,
    doornumber,
    carbody,
    drivewheel,
    enginelocation,
    wheelbase,
    carlength,
    carwidth,
    carheight,
    curbweight,
    enginetype,
    cylindernumber,
    enginesize,
    fuelsystem,
    boreratio,
    horsepower,
    citympg,
    highwaympg,
):
    new_data = [
        car.lower(),
        fueltype.lower(),
        asp[aspiration],
        dn[doornumber],
        carbody,
        drivew[drivewheel],
        el[enginelocation],
        wheelbase,
        carlength,
        carwidth,
        carheight,
        curbweight,
        enginetype,
        cylnum[cylindernumber],
        enginesize,
        fuelsystem,
        boreratio,
        horsepower,
        citympg,
        highwaympg,
    ]
    return transform(new_data)


car = gr.Dropdown(label="汽车品牌", choices=carNameCap)
fueltype = gr.Radio(label="燃料类型", choices=fuelCap)
aspiration = gr.Radio(label="吸气方式", choices=["标准", "涡轮"])
doornumber = gr.Radio(label="车门数量", choices=["双门", "四门"])
carbody = gr.Dropdown(label="车身类型", choices=carb)
drivewheel = gr.Radio(label="驱动轮", choices=["后驱", "前驱", "四驱"])
enginelocation = gr.Radio(label="发动机位置", choices=["前置", "后置"])
wheelbase = gr.Slider(label="汽车侧面车轮之间的距离（英寸）", minimum=50, maximum=300)
carlength = gr.Slider(label="汽车长度（英寸）", minimum=50, maximum=300)
carwidth = gr.Slider(label="汽车宽度（英寸）", minimum=50, maximum=300)
carheight = gr.Slider(label="汽车高度（英寸）", minimum=50, maximum=300)
curbweight = gr.Slider(label="汽车重量（磅）", minimum=500, maximum=6000)
enginetype = gr.Dropdown(label="发动机类型", choices=engtype)
cylindernumber = gr.Radio(label="气缸数量", choices=[2, 3, 4, 5, 6, 8, 12])
enginesize = gr.Slider(label="发动机尺寸（气缸内所有活塞的扫掠体积）", minimum=50, maximum=500)
fuelsystem = gr.Dropdown(label="燃油系统（连接到燃料: ", choices=fuelsys)
boreratio = gr.Slider(label="缸径比（气缸缸径与活塞行程之比）", minimum=1, maximum=6)
horsepower = gr.Slider(label="汽车的马力", minimum=25, maximum=400)
citympg = gr.Slider(label="城市里程（公里）", minimum=0, maximum=100)
highwaympg = gr.Slider(label="公路里程（km）", minimum=0, maximum=100)
Output = gr.Textbox()
app = gr.Interface(
    title="根据汽车规格预测二手汽车价格",
    fn=predict_price,
    inputs=[
        car,
        fueltype,
        aspiration,
        doornumber,
        carbody,
        drivewheel,
        enginelocation,
        wheelbase,
        carlength,
        carwidth,
        carheight,
        curbweight,
        enginetype,
        cylindernumber,
        enginesize,
        fuelsystem,
        boreratio,
        horsepower,
        citympg,
        highwaympg,
    ],
    outputs=Output,
)

app.launch(share=True)
