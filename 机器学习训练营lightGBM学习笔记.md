此笔记为阿里云天池机器学习训练营笔记，学习地址：https://tianchi.aliyun.com/s/1fc36a7e103eb9948c974f638e83a83b
感谢教程及阿里云提供平台

### 1.学习内容概括

**lightGBM是什么**

LightGBM是2017年由微软推出的可扩展机器学习系统，是微软旗下DMKT的一个开源项目，由2014年首届阿里巴巴大数据竞赛获胜者之一柯国霖老师带领开发。它是一款基于GBDT（梯度提升决策树）算法的分布式梯度提升框架，为了满足缩短模型计算时间的需求，LightGBM的设计思路主要集中在减小数据对内存与计算性能的使用，以及减少多机器并行计算时的通讯代价。

LightGBM可以看作是XGBoost的升级豪华版，在获得与XGBoost近似精度的同时，又提供了更快的训练速度与更少的内存消耗。正如其名字中的Light所蕴含的那样，LightGBM在大规模数据集上跑起来更加优雅轻盈，一经推出便成为各种数据竞赛中刷榜夺冠的神兵利器。

**优点：**

1. 简单易用。提供了主流的Python\C++\R语言接口，用户可以轻松使用LightGBM建模并获得相当不错的效果。
2. 高效可扩展。在处理大规模数据集时高效迅速、高准确度，对内存等硬件资源要求不高。
3. 鲁棒性强。相较于深度学习模型不需要精细调参便能取得近似的效果。
4. LightGBM直接支持缺失值与类别特征，无需对数据额外进行特殊处理

**缺点：**

LightGBM的主要缺点：

1. 相对于深度学习模型无法对时空位置建模，不能很好地捕获图像、语音、文本等高维数据。
2. 在拥有海量训练数据，并能找到合适的深度学习模型时，深度学习的精度可以遥遥领先LightGBM。

### 2.学习内容

#### 学习目标

- 了解 LightGBM 的参数与相关知识
- 掌握 LightGBM 的Python调用并将其运用到英雄联盟游戏胜负预测数据集上

#### lightGBM使用基本步骤

- Step1: 库函数导入
- Step2: 数据读取/载入
- Step3: 数据信息简单查看
- Step4: 可视化描述
- Step5: 利用 LightGBM 进行训练与预测
- Step6: 利用 LightGBM 进行特征选择
- Step7: 通过调整参数获得更好的效果

#### lightGBM原理

LightGBM底层实现了GBDT算法，并且添加了一系列的新特性：

1. 基于直方图算法进行优化，使数据存储更加方便、运算更快、鲁棒性强、模型更加稳定等。
2. 提出了带深度限制的 Leaf-wise 算法，抛弃了大多数GBDT工具使用的按层生长 (level-wise) 的决策树生长策略，而使用了带有深度限制的按叶子生长策略，可以降低误差，得到更好的精度。
3. 提出了单边梯度采样算法，排除大部分小梯度的样本，仅用剩下的样本计算信息增益，它是一种在减少数据量和保证精度上平衡的算法。
4. 提出了互斥特征捆绑算法，高维度的数据往往是稀疏的，这种稀疏性启发我们设计一种无损的方法来减少特征的维度。通常被捆绑的特征都是互斥的（即特征不会同时为非零值，像one-hot），这样两个特征捆绑起来就不会丢失信息。

LightGBM是基于CART树的集成模型，它的思想是串联多个决策树模型共同进行决策。

LightGBM采用迭代预测误差的方法串联。举个通俗的例子，我们现在需要预测一辆车价值3000元。我们构建决策树1训练后预测为2600元，我们发现有400元的误差，那么决策树2的训练目标为400元，但决策树2的预测结果为350元，还存在50元的误差就交给第三棵树……以此类推，每一颗树用来估计之前所有树的误差，最后所有树预测结果的求和就是最终预测结果

#### 基本参数调整

1. **num_leaves参数** 这是控制树模型复杂度的主要参数，一般的我们会使num_leaves小于（2的max_depth次方），以防止过拟合。由于LightGBM是leaf-wise建树与XGBoost的depth-wise建树方法不同，num_leaves比depth有更大的作用。、
2. **min_data_in_leaf** 这是处理过拟合问题中一个非常重要的参数. 它的值取决于训练数据的样本个树和 num_leaves参数. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合. 实际应用中, 对于大数据集, 设置其为几百或几千就足够了.
3. **max_depth** 树的深度，depth 的概念在 leaf-wise 树中并没有多大作用, 因为并不存在一个从 leaves 到 depth 的合理映射。

#### 针对训练速度的参数调整

1. 通过设置 **bagging_fraction** 和 **bagging_freq** 参数来使用 bagging 方法。
2. 通过设置 **feature_fraction** 参数来使用特征的子抽样。
3. 选择较小的 **max_bin** 参数。
4. 使用 **save_binary** 在未来的学习过程对数据加载进行加速。

#### 针对准确率的参数调整

1. 使用较大的 **max_bin** （学习速度可能变慢）
2. 使用较小的 **learning_rate** 和较大的 num_iterations
3. 使用较大的 **num_leaves** （可能导致过拟合）
4. 使用更大的训练数据
5. 尝试 **dart** 模式

#### 针对过拟合的参数调整

1. 使用较小的 **max_bin**
2. 使用较小的 **num_leaves**
3. 使用 **min_data_in_leaf** 和 **min_sum_hessian_in_leaf**
4. 通过设置 **bagging_fraction** 和 **bagging_freq** 来使用 bagging
5. 通过设置 **feature_fraction** 来使用特征子抽样
6. 使用更大的训练数据
7. 使用 **lambda_l1**, **lambda_l2** 和 **min_gain_to_split** 来使用正则
8. 尝试 **max_depth** 来避免生成过深的树

#### 调用方法

直接导入lightGBM包进行使用

```python
## 导入LightGBM模型
from lightgbm.sklearn import LGBMClassifier
## 定义 LightGBM 模型 
clf = LGBMClassifier()
# 在训练集上训练LightGBM模型
clf.fit(x_train, y_train)
```

预测并输出混淆矩阵

```python
## 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
from sklearn import metrics

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
```

#### 特征选择

**查看属性feature_importances_**

```python
sns.barplot(y=data_features_part.columns, x=clf.feature_importances_)
```

初次之外，我们还可以使用LightGBM中的下列重要属性来评估特征的重要性。

- gain:当利用特征做划分的时候的评价基尼指数
- split:是以特征用到的次数来评价

```python
from sklearn.metrics import accuracy_score
from lightgbm import plot_importance

def estimate(model,data):

    #sns.barplot(data.columns,model.feature_importances_)
    ax1=plot_importance(model,importance_type="gain")
    ax1.set_title('gain')
    ax2=plot_importance(model, importance_type="split")
    ax2.set_title('split')
    plt.show()
def classes(data,label,test):
    model=LGBMClassifier()
    model.fit(data,label)
    ans=model.predict(test)
    estimate(model, data)
    return ans
 
ans=classes(x_train,y_train,x_test)
pre=accuracy_score(y_test, ans)
print('acc=',accuracy_score(y_test,ans))

```

#### 网格搜索

网格搜索（Grid Search）名字非常大气，但是用简答的话来说就是你手动的给出一个模型中你想要改动的所用的参数，程序自动的帮你使用穷举法来将所用的参数都运行一遍。**决策树**中我们常常将最大树深作为需要调节的参数

```python
## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV

## 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
feature_fraction = [0.5, 0.8, 1]
num_leaves = [16, 32, 64]
max_depth = [-1,3,5,8]

parameters = { 'learning_rate': learning_rate,
              'feature_fraction':feature_fraction,
              'num_leaves': num_leaves,
              'max_depth': max_depth}
model = LGBMClassifier(n_estimators = 50)

## 进行网格搜索
clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=3, n_jobs=-1)
clf = clf.fit(x_train, y_train)
```

搜索结束后调用最优参数

```python
## 网格搜索后的最好参数为

clf.best_params_
```



### 3.问题与解决

**1.小提琴图如何使用？**

小提琴是四分位图的升级版

**●（1）**小提琴图的“胖肚子”可以显示出数据分布特征，肚子越胖，数据越集中。四分位图在这一点上表现不如小提琴图。

**●（2）**小提琴中的散点代表每一个个体数据。假如组内数据很多，采用单纯散点呈现会很杂乱。此时选择小提琴图模式，会更加清晰。

**●（3）**小提琴中的几条横线代表四分位数，其中蓝色线条代表的是中位数。因此采用小提琴模式表达非正态数据既适合也美观。

**2.什么是Leaf-wise**

遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，实际上很多叶子的分裂增益较低，没必要进行搜索和分裂，因此带来了很多没必要的计算开销。

![img](https://pic4.zhimg.com/80/v2-79a074ec2964a82301209fb66df37113_720w.jpg)

LightGBM采用Leaf-wise的增长策略，该策略每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，Leaf-wise的优点是：在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度；Leaf-wise的缺点是：可能会长出比较深的决策树，产生过拟合。因此LightGBM会在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

![img](https://pic2.zhimg.com/80/v2-e762a7e4c0366003d7f82e1817da9f89_720w.jpg)

**3.Boosting和Bagging的区别**

|       | bagging |boosting|
| ------------ | :------: | :----------------------------------------------------------: |
| 样本选择     | 放回抽样 | 全量选取                                                     |
| 样本权重     | 权重相等 | 错误率大的样本权重越大                                       |
| 弱学习器权重 | 权重相等 | 准确率大的分类器权重越大                                     |
| 并行计算     | 可以并行 | 不能并行，各个分类器按顺序生成，后一个模型参数要依赖前面的模型结果 |

### 4.归纳与总结

 boosting 算法在训练样本量有限、所需训练时间较短、缺乏调参知识等场景依然有其不可或缺的优势。lightGBM相对比XGBoost,拥有更快的训练效率，低内存使用，更高的准确率，支持并行化学习，可以处理大规模数据。确实是在比赛中非常好用的一种模型，调参方法也比xgboost更加合理，使用网格搜索能够快速找到最优参数，未来还需要多加学习
