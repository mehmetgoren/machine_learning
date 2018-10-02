"""
    seaborn is a kind of wrapper for matplotlib.pylot
"""

import matplotlib.pyplot as plt
import seaborn as sb

tips = sb.load_dataset("tips")
datas_tips = tips.head()


sb.distplot(tips["total_bill"])
sb.distplot(tips["total_bill"], kde=False, bins=40)





sb.jointplot(data=tips, x="total_bill", y="tip")

sb.jointplot(data=tips, x="total_bill", y="tip", kind="hex")

sb.jointplot(data=tips, x="total_bill", y="tip", kind="reg")

sb.jointplot(data=tips, x="total_bill", y="tip", kind="kde")





sb.pairplot(tips)

sb.pairplot(tips, hue="sex")





sb.rugplot(tips["total_bill"])





sb.barplot(data=tips, x="sex", y="total_bill")#erkeke, kadın bahşiş dağılımı.

import numpy as np
sb.barplot(data=tips, x="sex", y="total_bill", estimator=np.std)#x = kategorical






sb.countplot(x="sex", data=tips)

sb.boxplot(x="day", y="total_bill", data=tips)#outliar bulur
sb.boxplot(x="day", y="total_bill", data=tips, hue="smoker")


sb.violinplot(x="day", y="total_bill", data=tips,hue="sex",split=True)

sb.stripplot(x="day", y="total_bill", data=tips,hue="sex", split=True, jitter=True)


sb.swarmplot(x="day", y="total_bill", data=tips,hue="sex", split=True, jitter=True)


sb.factorplot(data=tips, x="day",y="total_bill", kind="bar")#bu da hepsinini karışımı.





flights = sb.load_dataset("flights")
datas_flights  = flights.head()
tips_corr = tips.corr()
sb.heatmap(tips_corr, annot=True, cmap="coolwarm")

pivot_table = flights.pivot_table(index="month",columns="year", values="passengers")

sb.clustermap(pivot_table, cmap="coolwarm")





iris = sb.load_dataset("iris")
iris_head = iris.head()
g = sb.pairplot(iris)
#g.map(plt.scatter)
g.map_diag(sb.distplot)
g.map_upper(plt.scatter)
g.map_lower(sb.kdeplot)





lienar regressions
sb.lmplot(x="total_bill", y="tip", data=tips, hue="sex", markers=["o","v"])




plt.figure(figsize=(12,3))
sb.set_style("darkgrid")
sb.countplot(x="sex", data=tips)