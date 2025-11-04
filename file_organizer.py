


#数据基本处理
# def dm01_():
#     churn_pd = pd.read_csv(r'D:\workspace\ai_ML_bj\PythonProject\day04线性回归\逻辑回归\churn.csv')
#     print(f'data.info-->\n{churn_pd.info}')
#
#     #1.处理类别型的额数据 类别型数据做one-shot编码
#     churn_pd = pd.get_dummies(churn_pd)
#     print(churn_pd.info)
#
#     #2。去除列
#     churn_pd = churn_pd.drop(['Churn_No', 'gender_Male'], axis=1)
#
#
#     #3.列标签重命名，打印列明
#     print('churn_pd.columns',churn_pd.columns)
#     churn_pd = churn_pd.rename(columns={'Churn_Yes': 'flag'})
#     print('churn_pd.columns', churn_pd.columns)
#
#     #4.查看标签的分布情况0.26用户流失
#     value_counts = churn_pd.flag.value_counts(1)
#     print('value_counts-->\n', value_counts)
#     print('从标签的分类中可以看出: 属于标签分类不平衡样本')




#
# if __name__ == '__main__':
#     dm01_()
#
#


#导入依赖包
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,roc_auc_score,classification_report,accuracy_score

#2.数据处理
#2.1读取数据
data = pd.read_csv(r'D:\workspace\ai_ML_bj\PythonProject\day04线性回归\逻辑回归\churn.csv')
print(f'data.info-->\n{data.info()}')
print(f'data.describe()-->\n{data.describe()}')
print(f'data.head()-->\n{data.head()}')

#2.2中文内容 one-shot编码
dum_data = pd.get_dummies(data)
print(f'dum_data-->\n{dum_data}')
print(f'dum_data.columns-->\n{dum_data.columns}')

#2.3删除两列
data = dum_data.drop(['gender_Male','Churn_No'], axis=1)

#2.4重命名
data = data.rename(columns={'Churn_Yes': 'flag'})
print(data.head())
print(data.flag.value_counts())


sns.countplot(data=data, y='Contract_Month', hue='flag')
plt.show()

#3.特征工程
#3.1取数据和标签
x = data[['Contract_Month','Contract_1YR','PaymentElectronic','PaymentBank','internet_other']]
y = data['flag']

#3.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#4.模型训练
model = LogisticRegression()
model.fit(x_train, y_train)

#5.模型预测
y_pred = model.predict(x_test)
print(y_pred)

#6.模型评估
#6.1ACC
acc_score = model.score(x_test, y_test)
acc_1 = accuracy_score(y_test, y_pred)

print(f'acc_score-->{acc_score}')

#6.2AUC_ROC
roc_auc_score = roc_auc_score(y_test, y_pred)
print(f'roc_auc_score-->{roc_auc_score}')

#6.3 clas_report
report = classification_report(y_test, y_pred)
print(f'report-->{report}')