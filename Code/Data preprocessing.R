library(Hmisc)
library(rms)
library(ggplot2)
library(patchwork)
library(MatchIt)  #Load matchit
library(dplyr)
library("autoReg")
library(finalfit)
library(flextable)
library(ggpubr)
library(devtools)
library(customLayout)
library(pROC) # 加载pROC包
library(cutoff)
library(cowplot)
library(rcssci)
library(ggrcs)
library(carData)
library(car)
library(Rmisc)
library(qqplotr)
library(dbscan)
library(cluster)
# library(statnet)
library(circlize)
library(reshape2)
library(ggradar)
library(fmsb)  # 加载fmsb包
library(grid)
library(survival)
library(survminer)
library(verification)
rm(list = ls()); options(stringsAsFactors = F); options(warn = -1); options(digits=3) # 3位小数
setwd("E:\\MIMIC\\CCI\\result\\标准化\\论文浓缩版");getwd()

###-----Missing Date-------###
data_mimic <- read.csv("./data/sepsis3_cci.csv")
colnames(data_mimic)
x_mimic <- data_mimic[,c(18:128)]

# colnames(x_mimic) <- c(`Mean HR`,`Minimum HR`,`Maximum HR`,`Mean SBP`,`Minimum SBP`,`Maximum SBP`,`Mean DBP`,
#                        `Minimum DBP`,`Maximum DBP`,`Mean MAP`,`Minimum MAP`,`Maximum MAP`,`Mean RR`,`Minimum RR`,
#                        `Maximum RR`,`Mean Temperature`,`Minimum Temperature`,`Maximum Temperature`,`Mean PO2`,
#                        `Minimum PO2`,`Maximum PO2`,`Mean PCO2`,`Minimum PCO2`,`Maximum PCO2`,`Mean FiO2`,`Minimum FiO2`,
#                        `Maximum FiO2`,`Mean AaDO2`,`Minimum AaDO2`,`Maximum AaDO2`,`Mean Base Excess`,
#                        `Minimum Base Excess`,`Maximum Base Excess`,`Mean RBC`,`Minimum RBC`,`Maximum RBC`,
#                        `Mean Hemoglobin`,`Minimum Hemoglobin`,`Maximum Hemoglobin`,`Mean Hematocrit`,
#                        `Minimum Hematocrit`,`Maximum Hematocrit`,`Mean RDW`,`Minimum RDW`,`Maximum RDW`,`Mean MCH`,
#                        `Minimum MCH`,`Maximum MCH`,`Mean MCV`,`Minimum MCV`,`Maximum MCV`,`Mean MCHC`,`Minimum MCHC`,
#                        `Maximum MCHC`,`Mean Platelet`,`Minimum Platelet`,`Maximum Platelet`,`Mean WBC`,`Minimum WBC`,
#                        `Maximum WBC`,`Mean Basophils`,`Minimum Basophils`,`Maximum Basophils`,`Mean Eosinophils`,
#                        `Minimum Eosinophils`,`Maximum Eosinophils`,`Mean Lymphocytes`,`Minimum Lymphocytes`,
#                        `Maximum Lymphocytes`,`Mean Monocytes`,`Minimum Monocytes`,`Maximum Monocytes`,
#                        `Mean Neutrophils`,`Minimum Neutrophils`,`Maximum Neutrophils`,`Mean Albumin`,
#                        `Minimum Albumin`,`Maximum Albumin`,`Mean AG`,`Minimum AG`,`Maximum AG`,`Mean BUN`,
#                        `Minimum BUN`,`Maximum BUN`,`Mean Calcium`,`Minimum Calcium`,`Maximum Calcium`,
#                        `Mean Chloride`,`Minimum Chloride`,`Maximum Chloride`,`Mean Sodium`,`Minimum Sodium`,
#                        `Maximum Sodium`,`Mean Potassium`,`Minimum Potassium`,`Maximum Potassium`,`Mean Glucose`,
#                        `Minimum Glucose`,`Maximum Glucose`,`Mean Creatinine`,`Minimum Creatinine`,
#                        `Maximum Creatinine`,`Mean INR`,`Minimum INR`,`Maximum INR`,`Mean PT`,`Minimum PT`,
#                        `Maximum PT`,`Mean PTT`,`Minimum PTT`,`Maximum PTT`
# )
xx_mimic <- data.frame(
  X <- gsub('[.]', ' ', colnames(x_mimic)))

for (i in c(1:111)){
  xx_mimic[i,'Null'] <- sum(is.na(x_mimic[,i]))/length(x_mimic[,i])
  if (xx_mimic[i,'Null']<0.1){
    xx_mimic[i,'Group'] <- 'Good'
  }
  if (xx_mimic[i,'Null']>=0.1 && xx_mimic[i,'Null']<0.3){
    xx_mimic[i,'Group'] <- 'Normal'
  }
  if (xx_mimic[i,'Null']>=0.3 && xx_mimic[i,'Null']<0.5){
    xx_mimic[i,'Group'] <- 'Bad'
  }
  if (xx_mimic[i,'Null']>=0.5){
    xx_mimic[i,'Group'] <- 'Remove'
  }
}
xx_mimic[,1] <- 
sorted_df <- xx_mimic[order(xx_mimic$Null,decreasing = F), ]
sorted_df$idx <- c(1:111)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_A <- ggplot(data = sorted_df, aes(x = X, y = Null,fill=Group)) +
  geom_bar(stat = "identity", 
           width = 0.7, size = 0.25, alpha = 1) +
  scale_color_manual(values=c("skyblue", "maroon", "gold", "brown1"))+
  ylim(0, 1) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 5, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("MIMIC")+
  theme(legend.position="none") #隐藏图例

data_eicu <- read.csv("./data/eicu_cci.csv")
colnames(data_eicu)
x_eicu <- data_eicu[,c(22:129)]
# colname(x_eicu) <- c(`Mean HR`,`Minimum HR`,`Maximum HR`,`Mean SBP`,`Minimum SBP`,`Maximum SBP`,`Mean DBP`,
#                      `Minimum DBP`,`Maximum DBP`,`Mean MAP`,`Minimum MAP`,`Maximum MAP`,`Mean RR`,`Minimum RR`,
#                      `Maximum RR`,`Mean Temperature`,`Minimum Temperature`,`Maximum Temperature`,`Mean PO2`,
#                      `Minimum PO2`,`Maximum PO2`,`Mean PCO2`,`Minimum PCO2`,`Maximum PCO2`,`Mean FiO2`,`Minimum FiO2`,
#                      `Maximum FiO2`,`Mean Base Excess`,
#                      `Minimum Base Excess`,`Maximum Base Excess`,`Mean RBC`,`Minimum RBC`,`Maximum RBC`,
#                      `Mean Hemoglobin`,`Minimum Hemoglobin`,`Maximum Hemoglobin`,`Mean Hematocrit`,
#                      `Minimum Hematocrit`,`Maximum Hematocrit`,`Mean RDW`,`Minimum RDW`,`Maximum RDW`,`Mean MCH`,
#                      `Minimum MCH`,`Maximum MCH`,`Mean MCV`,`Minimum MCV`,`Maximum MCV`,`Mean MCHC`,`Minimum MCHC`,
#                      `Maximum MCHC`,`Mean Platelet`,`Minimum Platelet`,`Maximum Platelet`,`Mean WBC`,`Minimum WBC`,
#                      `Maximum WBC`,`Mean Basophils`,`Minimum Basophils`,`Maximum Basophils`,`Mean Eosinophils`,
#                      `Minimum Eosinophils`,`Maximum Eosinophils`,`Mean Lymphocytes`,`Minimum Lymphocytes`,
#                      `Maximum Lymphocytes`,`Mean Monocytes`,`Minimum Monocytes`,`Maximum Monocytes`,
#                      `Mean Neutrophils`,`Minimum Neutrophils`,`Maximum Neutrophils`,`Mean Albumin`,
#                      `Minimum Albumin`,`Maximum Albumin`,`Mean AG`,`Minimum AG`,`Maximum AG`,`Mean BUN`,
#                      `Minimum BUN`,`Maximum BUN`,`Mean Calcium`,`Minimum Calcium`,`Maximum Calcium`,
#                      `Mean Chloride`,`Minimum Chloride`,`Maximum Chloride`,`Mean Sodium`,`Minimum Sodium`,
#                      `Maximum Sodium`,`Mean Potassium`,`Minimum Potassium`,`Maximum Potassium`,`Mean Glucose`,
#                      `Minimum Glucose`,`Maximum Glucose`,`Mean Creatinine`,`Minimum Creatinine`,
#                      `Maximum Creatinine`,`Mean INR`,`Minimum INR`,`Maximum INR`,`Mean PT`,`Minimum PT`,
#                      `Maximum PT`,`Mean PTT`,`Minimum PTT`,`Maximum PTT`
# )
xx_eicu <- data.frame(
  X <- gsub('[.]', ' ', colnames(x_eicu)))
for (i in c(1:108)){
  xx_eicu[i,'Null'] <- sum(is.na(data_eicu[,i]))/length(data_eicu[,i])
  if (xx_eicu[i,'Null']<0.1){
    xx_eicu[i,'Group'] <- 'Good'
  }
  if (xx_eicu[i,'Null']>=0.1 && xx_eicu[i,'Null']<0.3){
    xx_eicu[i,'Group'] <- 'Normal'
  }
  if (xx_eicu[i,'Null']>=0.3 && xx_eicu[i,'Null']<0.5){
    xx_eicu[i,'Group'] <- 'Bad'
  }
  if (xx_eicu[i,'Null']>=0.5){
    xx_eicu[i,'Group'] <- 'Remove'
  }
}
sorted_df <- xx_eicu[order(xx_eicu$Null,decreasing = F), ]
sorted_df$idx <- c(1:108)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_B <- ggplot(data = sorted_df, aes(x = X, y = Null,fill=Group)) +
  geom_bar(stat = "identity", 
           width = 0.7, size = 0.25, alpha = 1) +
  scale_color_manual(values=c("skyblue", "maroon", "gold", "brown1"))+
  ylim(0, 1) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 5, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("eICU")
p_B
f1 <- cowplot::plot_grid(p_A, p_B,ncol = 2)
f1 <- p_A+p_B
ggsave(
  filename = paste("./result/sub/缺失值.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 10,             # 宽
  height = 10,            # 高
  units = "in",          # 单位
  dpi = 600              # 分辨率DPI
)

#----Missing value interpolation----#
# Missing value interpolation see miceforest.py

#----correlation coefficient----#
data <- read.csv("./data/sepsis3_cci_mice_drop.csv")[,c(18:125)];
colname <- colnames(data)

#检验变量间相关性
data_cor <- cor(scale(data),method = "spearman")
data_cor <- data.frame(data_cor)
colnames(data_cor) <- gsub('[.]', ' ', colnames(data_cor))
rownames(data_cor) <- gsub('[.]', ' ', rownames(data_cor))
library("pheatmap")
for (i in (1:108)){
  data_cor[i,i] <- 0
}
pheatmap(as.matrix(data_cor), scale = "none", #表示进行均一化的方向，值为 “row”, “column” 或者"none"
         cluster_rows = F,cluster_cols = F,show_rownames = T, show_colnames = T,
         fontsize =10, fontsize_row = 4, fontsize_col = 4, #热图中字体大小、行、列名字体大小
         color = colorRampPalette(c("#0000ff","#ffffff","#ff0000"))(100),
         annotation_row = NA, annotation_col = NA, #表示是否对行、列进行注释，默认NA
         annotation = NA, annotation_colors = NA,main = paste("MIMIC"),
         filename = paste("./result/sub/heatmap_MIMIC.pdf"))


df_cor <- data.frame(
  '变量1' = c(1),
  '变量2' = c(1),
  '相关系数' = c(1)
)
n <- 1
for (i in c(1:107)){
  for (j in c((i+1):108)){
    if (data_cor[i,j]>=0.8){
      df_cor[n,'变量1'] <- colnames(data_cor)[i]
      df_cor[n,'变量2'] <- rownames(data_cor)[j]
      df_cor[n,'相关系数'] <- data_cor[i,j]
      n <- n+1
    }
  }
}
write.csv(df_cor, file = "./result/sub/相关系数(80%).csv", row.names = FALSE)

# 导入igraph包
library(igraph)
# 创建网络关系数据框

relations  <- data.frame(from=df_cor$`变量1`, 
                    to=df_cor$`变量2`,
                    same.dept=rep(T,71),
                    friendship=df_cor[,3]*10, advice=df_cor[,3]
                    )
g <- graph_from_data_frame(relations, directed=F)
# print(g, e = TRUE, v = TRUE)
pdf("./result/sub/相关系数.pdf")
plot(g)
plot(g, edge.size=0.8, vertex.size=10,edge.curved=TRUE, vertex.color="white",edge.color="black")
dev.off()

install.packages("visNetwork")
library(visNetwork)
library(igraph)

# 计算相关性矩阵
cor_mat <- data_cor

# 将相关性矩阵转换为边列表（igraph对象）
graph <- graph_from_adjacency_matrix(cor_mat, mode = "undirected", weighted = TRUE, diag = FALSE)

# 将igraph对象转换为visNetwork对象
vis_net <- visIgraph(graph)

# 自定义图的外观
vis_net <- vis_net %>% 
  visEdges(arrows = 'to') %>%
  visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
  visLayout(randomSeed = 123)

# 绘制网络图
vis_net

data <- read.csv("./data/sepsis3_cci_eicu_drop2.csv")[,c(22:129)];
colname <- colnames(data)

#检验变量间相关性
data_cor <- cor(scale(data),method = "spearman")
data_cor <- data.frame(data_cor)
colnames(data_cor) <- gsub('[.]', ' ', colnames(data_cor))
rownames(data_cor) <- gsub('[.]', ' ', rownames(data_cor))
library("pheatmap")
for (i in (1:108)){
  data_cor[i,i] <- 0
}
pheatmap(as.matrix(data_cor), scale = "none", #表示进行均一化的方向，值为 “row”, “column” 或者"none"
         cluster_rows = F,cluster_cols = F,show_rownames = T, show_colnames = T,
         fontsize =10, fontsize_row = 4, fontsize_col = 4, #热图中字体大小、行、列名字体大小
         color = colorRampPalette(c("#0000ff","#ffffff","#ff0000"))(100),
         annotation_row = NA, annotation_col = NA, #表示是否对行、列进行注释，默认NA
         annotation = NA, annotation_colors = NA,main = paste("eICU"),
         filename = paste("./result/sub/heatmap_EICU.pdf"))

#---Kolmogorov-Smirnov test----#
data <- read.csv("./data/xg2.csv")
x = data[,c(18:78,8,103:116,98:99,101:102)]
data_eicu <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv")
x_eicu = data_eicu[,c(22:82,14,105:118,101:104)]
summary(x_eicu)

df_ks <- data.frame(
  '变量' = colnames(x_eicu)
)

for (i in c(1:80)){
  dg <- x[,i]
  l <- ks.test(scale(dg),"pnorm") #scale标准化后再检验
  df_ks[i,'Kolmogorov-Smirnov test MIMIC'] <- round(l$statistic,4)
  df_ks[i,'P-values MIMIC'] <- l$p.value
  dg2 <- x_eicu[,i]
  l2 <- ks.test(scale(dg2),"pnorm") #scale标准化后再检验
  df_ks[i,'Kolmogorov-Smirnov test EICU'] <- round(l2$statistic,4)
  df_ks[i,'P-values EICU'] <- l2$p.value
}
write.csv(df_ks, file = "./result/sub/K-S检验.csv", row.names = FALSE)

#---Quantile-Quantile Plot----#
data <- read.csv("./data/sepsis3_cci_eicu_drop2.csv"); # EICU
data <- data[,c(22:129)]
# data <- read.csv("./data/sepsis3_cci_mice_drop.csv") # MIMIC
# data <- data[,c(18:125)]
colnames(data) <- gsub('[.]', ' ', colnames(data))
p <- list()
n <- 1
for (i in c(1:108)){
  datap <- data.frame(
    index <- c(1:length(data[,i]))
  )
  datap$norm <- scale(data[,i])
  p[[n]] <-ggplot(data = datap, mapping = aes(sample = norm)) +
    stat_qq_band() +
    stat_qq_line(size = 0.2) +
    stat_qq_point(size = 0.001) +
    xlab("Theoretical")+
    ylab("Sample")+
    ggtitle(paste0(colnames(data)[i]))+
    theme_bw() + 
    theme(
      plot.title = element_text(family = "serif", #字体
                                face = "bold",     #字体加粗
                                color = "black",      #字体颜色
                                size = 6,          #字体大小
                                hjust = 0.5,          #字体左右的位置
                                vjust = 0.5,          #字体上下的高度
                                angle = 0),             #字体倾斜的角度
      axis.title = element_text(size = 4, face = "plain", color = "black"), # 设置标题的字体及大小
      axis.text = element_text(size = 3, face = "plain", color = "black") # 设置坐标轴的字体及大小
    )
  
  n <- n+1
}

f1 <- wrap_plots(p, ncol = 8)
ggsave(
  filename = paste("./result/sub/QQ图_eicu.tiff"), # EICU
  # filename = paste("./result/sub/QQ图_mimic.tiff"), # MIMIC
  plot = f1,
  width = 10,             # 宽
  height = 14,            # 高
  units = "in",          # 单位
  dpi = 600              # 分辨率DPI
)