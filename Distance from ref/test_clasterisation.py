import numpy as np
import rpy2
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()

importr('usedist')
importr('dynamicTreeCut')

rpy2.robjects.r("""
hclusterisation <- function (data){
  custom_dist <- function (dot1,dot2,threshold=2,penalty=5){
    delta <- abs(dot1[1]-dot2[1])
    if (dot1[2] == dot2[2] & delta <= threshold) {
      return(delta+penalty)
    }
    else {
      return(delta)
    }
  }
  distances <- dist_make(data,custom_dist,2,5)
  tree <- hclust(distances,method = "complete", members = NULL)
  clusters <- cutreeHybrid(tree,as.matrix(distances),
                           cutHeight = NULL,
                           minClusterSize = 3,
                           deepSplit = 1,
                           pamStage = TRUE)
  return(clusters)
}
""")

data_mz = np.array([1, 2, 2, 2, 3, 3, 9, 10, 10, 11, 11, 12])
data_id = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

data = np.stack([data_mz, data_id], axis=1)

r_func = rpy2.robjects.globalenv['hclusterisation']
r_data = numpy2ri.py2rpy(data)
result = r_func(r_data)
labels = np.array(dict(result.items())['labels'])
max_num = labels.max()
dots = []
for i in range(1, max_num + 1):
    dots.append(data_mz[labels == i])
print(dots)
