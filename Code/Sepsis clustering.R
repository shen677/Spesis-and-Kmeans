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
library(scitb)
rm(list = ls()); options(stringsAsFactors = F); options(warn = -1); options(digits=3) # 3位小数
setwd("E:\\MIMIC\\CCI\\result\\标准化\\论文浓缩版");getwd()

###-----Hopkins statistical-------###
# Hopkins statistical values see kmeans.py

###-----reachability distance-------###
data <- read.csv("./data/xg2.csv"); # MIMIC
colnames(data) <- gsub('[.]', ' ', colnames(data))
x <- data[, c(18:78)];
# data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv"); #EICU
# colnames(data) <- gsub('[.]', ' ', colnames(data))
# x <- data[, c(22:82)];
for (i in c(1:61)){
  if (length(unique(x[,i]))>=10){
    mean_value <- mean(x[,i])
    sd_value <- sd(x[,i])
    x[x[,i]>(mean_value+20*sd_value),i] <- (mean_value+20*sd_value)
    x[x[,i]<(mean_value-20*sd_value),i] <- (mean_value-20*sd_value)
    
  }
}
x <- scale(x)
x <- data.frame(x)
x_optics <- optics(x,minPts = 1);
# pdf(file="./result/sub/可达距离_eicu.pdf", width=8, height=5, pointsize=8)
pdf(file="./result/sub/可达距离_mimic.pdf", width=8, height=5, pointsize=8)
plot(x_optics, cluster = TRUE, predecessor = FALSE)
dev.off()

###-----Consensus Kmeans clustering-------###
# Consensus Kmeans clustering see kmeans.py

###-----Feature Importance of Kmeans Clustering-------###
# See kmeans.py for details of "importance_feature_mimic.csv" and "importance_feature_eicu.csv"
data <- read.csv("./data/importance_feature_mimic.csv"); #MIMIC
# data <- read.csv("./data/importance_feature_eicu.csv");
for (i in c(1:61)){
  data[i,'X'] <- gsub('[.]', ' ', data[i,'X'])
}

sorted_df <- data[order(data$all,decreasing = T), ]
sorted_df$idx <- c(1:61)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
for ( i in c(2:6)){
  print(max(sorted_df[,i]))
  print(min(sorted_df[,i]))
}
p_all <- ggplot(data = sorted_df[1:20,], aes(x = X, y = all)) +
  geom_bar(stat = "identity", 
           width = 0.8, colour = "brown1", size = 0.25,
           fill = "brown1", alpha = 1) +
  ylim(0, ceiling(max(sorted_df$all)/0.005)*0.005) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 12, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("Overall")

sorted_df <- data[order(data$A,decreasing = T), ]
sorted_df$idx <- c(1:61)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_A <- ggplot(data = sorted_df[1:20,], aes(x = X, y = A)) +
  geom_bar(stat = "identity", 
           width = 0.8, colour = "cyan", size = 0.25,
           fill = "cyan", alpha = 1) +
  ylim(0, ceiling(max(sorted_df$A)/0.005)*0.005) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 12, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("Phenotype A")

sorted_df <- data[order(data$B,decreasing = T), ]
sorted_df$idx <- c(1:61)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_B <- ggplot(data = sorted_df[1:20,], aes(x = X, y = B)) +
  geom_bar(stat = "identity", 
           width = 0.8, colour = "maroon", size = 0.25,
           fill = "maroon", alpha = 1) +
  ylim(0, ceiling(max(sorted_df$B)/0.005)*0.005) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 12, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("Phenotype B")

sorted_df <- data[order(data$C,decreasing = T), ]
sorted_df$idx <- c(1:61)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_C <- ggplot(data = sorted_df[1:20,], aes(x = X, y = C)) +
  geom_bar(stat = "identity", 
           width = 0.8, colour = "orange", size = 0.25,
           fill = "orange", alpha = 1) +
  ylim(0, ceiling(max(sorted_df$C)/0.005)*0.005) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 12, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("Phenotype C")

sorted_df <- data[order(data$D,decreasing = T), ]
sorted_df$idx <- c(1:61)
sorted_df$X = factor(sorted_df$X, levels = sorted_df$X[order(-sorted_df$idx)])
p_D <- ggplot(data = sorted_df[1:20,], aes(x = X, y = D)) +
  geom_bar(stat = "identity", 
           width = 0.8, colour = "wheat", size = 0.25,
           fill = "wheat", alpha = 1) +
  ylim(0, ceiling(max(sorted_df$D)/0.005)*0.005) + # 设置y轴范围
  theme(
    axis.title = element_text(size = 15, face = "plain", color = "black"), # 设置标题的字体及大小
    axis.text = element_text(size = 12, face = "plain", color = "black") # 设置坐标轴的字体及大小
  )+
  coord_flip()+
  xlab(" ")+
  ylab("Phenotype D")
f1 <- cowplot::plot_grid(p_A,p_B,p_C,p_D,ncol = 2)#横向排列
ggsave(
  filename = paste("./result/sub/聚类重要性_mimic1.pdf"), # MIMIC
  # filename = paste("./result/sub/聚类重要性_eicu1.pdf"), # EICU
  plot = f1,
  width = 10,             # 宽
  height = 8,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

ggsave(
  filename = paste("./result/sub/聚类重要性_mimic2.pdf"), # MIMIC
  # filename = paste("./result/sub/聚类重要性_eicu2.pdf"), # EICU
  plot = p_all,
  width = 5,             # 宽
  height = 4,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

###-----Chordal Graph-------###
data <- read.csv("./data/变量与聚类的影响_mimic.csv"); #MIMIC
# data <- read.csv("./data/变量与聚类的影响_eicu.csv");
data_ <- data.frame(
  label = c("Vital Signs","Blood Gas Analysis","Blood Routine","Biochemistry","Coagulation")
)

data_[1,"A"] <- sum(data[1:18,"A"])
data_[1,"B"] <- sum(data[1:18,"B"])
data_[1,"C"] <- sum(data[1:18,"C"])
data_[1,"D"] <- sum(data[1:18,"D"])

data_[2,"A"] <- sum(data[19:29,"A"])
data_[2,"B"] <- sum(data[19:29,"B"])
data_[2,"C"] <- sum(data[19:29,"C"])
data_[2,"D"] <- sum(data[19:29,"D"])

data_[3,"A"] <- sum(data[30:46,"A"])
data_[3,"B"] <- sum(data[30:46,"B"])
data_[3,"C"] <- sum(data[30:46,"C"])
data_[3,"D"] <- sum(data[30:46,"D"])

data_[4,"A"] <- sum(data[47:58,"A"])
data_[4,"B"] <- sum(data[47:58,"B"])
data_[4,"C"] <- sum(data[47:58,"C"])
data_[4,"D"] <- sum(data[47:58,"D"])

data_[5,"A"] <- sum(data[59:61,"A"])
data_[5,"B"] <- sum(data[59:61,"B"])
data_[5,"C"] <- sum(data[59:61,"C"])
data_[5,"D"] <- sum(data[59:61,"D"])

my.data<-t(as.matrix(data_[,2:5])) # 矩阵化
# 手动设置行列名（可选）
rownames(my.data) <-c("Phenotype A", "Phenotype B", "Phenotype C", "Phenotype D")
colnames(my.data) <-c("Vital Signs","Blood Gas Analysis","Blood Routine","Biochemistry","Coagulation")

grid.col = NULL

# 定义处理的颜色，这里随便选取了4个颜色，大家可以根据自己的喜好制定好看的配色
grid.col[c("Phenotype A", "Phenotype B", "Phenotype C", "Phenotype D")] = c("cyan", "maroon", "orange", "wheat") 

# 定义微生物各个门的颜色，
grid.col[colnames(my.data)] = c("lavender", "khaki","mistyrose", "sienna1", "skyblue")# c("lavender","chocolate", "khaki","mistyrose", "sienna1", "skyblue", "brown1", "gold", "maroon", "salmon", "moccasin","wheat","black","green","cyan","pink","orange")

# 参数设置
circos.par(gap.degree = c(rep(2, nrow(my.data)-1), 10, rep(2, ncol(my.data)-1), 10),
           start.degree = 180)

pdf(file="./result/sub/弦图_mimic.pdf", width=8, height=5, pointsize=8)
# pdf(file="./result/sub/弦图_eicu.pdf", width=8, height=5, pointsize=8)
chordDiagram(my.data,
             directional = TRUE,
             diffHeight = 0.06,
             grid.col = grid.col, 
             transparency = 0.5) 
# 图例制作
legend("right",pch=20,legend=colnames(my.data),
       col=grid.col[colnames(my.data)],bty="n",
       cex=1,pt.cex=3,border="black") # 设定图例
dev.off()

###-----Comparison of features among sepsis subphenotypes after consensus Kmeans clustering-------###
# data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv");# EICU
# x = data[,22:82]
data <- read.csv("./data/xg2.csv"); # MIMIC
x = data[,18:123]
col <- colnames(x)

for (i in c(1:length(col))){
  col[i] <- gsub('[.]', ' ', col[i])
}
for (i in c(1:61)){
  if (length(unique(x[,i]))>=10){
    mean_value <- mean(x[,i])
    sd_value <- sd(x[,i])
    x[x[,i]>(mean_value+20*sd_value),i] <- (mean_value+20*sd_value)
    x[x[,i]<(mean_value-20*sd_value),i] <- (mean_value-20*sd_value)
    
  }
}
x <- scale(x)
x <- data.frame(x)
x_A <- x[data$kmeans==0,]
x_B <- x[data$kmeans==1,]
x_C <- x[data$kmeans==2,]
x_D <- x[data$kmeans==3,]
colnames(x) <- col
X <- data.frame(
  name = colnames(x)
)
for (i in c(1:61)){
  X[i,"A"] <- mean(x_A[,i])
  X[i,"B"] <- mean(x_B[,i])
  X[i,"C"] <- mean(x_C[,i])
  X[i,"D"] <- mean(x_D[,i])
}
windowsFonts(Times=windowsFont("Times New Roman"))
p <- list()
#--Vital Signs
mydata <- melt(X[c(1:18),],id="name")
colnames(mydata) <- c("name","Phenotype","value")
p[[1]] <- ggplot(data = mydata,aes(x=value,y=name,group = Phenotype,color=Phenotype,shape=Phenotype))+
  geom_line(orientation = "y", size=1.25)+
  #geom_point()+
  xlab("Standardized Mean")+#横坐标名称
  ylab("Vital Signs")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(#panel.grid.major=element_line(colour=NA),
    #panel.background = element_rect(fill = "transparent",colour = NA),
    #plot.background = element_rect(fill = "transparent",colour = NA),
    panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
    #text = element_text(family = "STXihei"),#设置中文字体的显示 STXihei
    #legend.position = c(1.015,.175),#更改图例的位置，放至图内部的左上角
    legend.box.background = element_rect(color="black"))+
  scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))+
  theme(legend.position="none") #隐藏图例
#--Blood Gas Analysis
mydata <- melt(X[c(19:29),],id="name")
colnames(mydata) <- c("name","Phenotype","value")
p[[2]] <- ggplot(data = mydata,aes(x=value,y=name,group = Phenotype,color=Phenotype,shape=Phenotype))+
  geom_line(orientation = "y", size=1.25)+
  #geom_point()+
  xlab("Standardized Mean")+#横坐标名称
  ylab("Blood Gas Analysis")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(#panel.grid.major=element_line(colour=NA),
    #panel.background = element_rect(fill = "transparent",colour = NA),
    #plot.background = element_rect(fill = "transparent",colour = NA),
    panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
    #text = element_text(family = "STXihei"),#设置中文字体的显示 STXihei
    #legend.position = c(1.015,.175),#更改图例的位置，放至图内部的左上角
    legend.box.background = element_rect(color="black"))+
  scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))+
  theme(legend.position="none") #隐藏图例
#--Blood Routine
mydata <- melt(X[c(30:46),],id="name")
colnames(mydata) <- c("name","Phenotype","value")
p[[3]] <- ggplot(data = mydata,aes(x=value,y=name,group = Phenotype,color=Phenotype,shape=Phenotype))+
  geom_line(orientation = "y", size=1.25)+
  #geom_point()+
  xlab("Standardized Mean")+#横坐标名称
  ylab("Blood Routine")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(#panel.grid.major=element_line(colour=NA),
    #panel.background = element_rect(fill = "transparent",colour = NA),
    #plot.background = element_rect(fill = "transparent",colour = NA),
    panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
    #text = element_text(family = "STXihei"),#设置中文字体的显示 STXihei
    #legend.position = c(1.015,.175),#更改图例的位置，放至图内部的左上角
    legend.box.background = element_rect(color="black"))+
  scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))+
  theme(legend.position="none") #隐藏图例
#--Biochemistry and Coagulation
mydata <- melt(X[c(47:61),],id="name")
colnames(mydata) <- c("name","Phenotype","value")
p[[4]] <- ggplot(data = mydata,aes(x=value,y=name,group = Phenotype,color=Phenotype,shape=Phenotype))+
  geom_line(orientation = "y", size=1.25)+
  #geom_point()+
  xlab("Standardized Mean")+#横坐标名称
  ylab("Biochemistry and Coagulation")+#纵坐标名称
  theme_bw() +#去掉背景灰色
  theme(#panel.grid.major=element_line(colour=NA),
    #panel.background = element_rect(fill = "transparent",colour = NA),
    #plot.background = element_rect(fill = "transparent",colour = NA),
    panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
    #text = element_text(family = "STXihei"),#设置中文字体的显示 STXihei
    #legend.position = c(1.015,.175),#更改图例的位置，放至图内部的左上角
    legend.box.background = element_rect(color="black"))+
  scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))

f <- wrap_plots(p, ncol = 4)
ggsave(
  filename = paste("./result/sub/61各变量特征_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  # filename = paste("./result/sub/61各变量特征_eicu.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f,
  width = 21,             # 宽
  height = 7,            # 高
  units = "in",          # 单位
  dpi = 500              # 分辨率DPI
)

###-----Radar Chart-------###
#-----Comorbidities------#
data <- read.csv("./data/xg2.csv"); # MIMC
x1 = data[,80:96]
x <- data.frame()
for (i in c(1:33177)){
  x[i,'Myocardial Infarct'] <- x1[i,1]
  x[i,'Congestive Heart Failure'] <- x1[i,2]
  x[i,'Chronic Pulmonary Disease'] <- x1[i,6]
  x[i,'Liver Disease'] <- max(x1[i,9],x1[i,15],na.rm=T)
  x[i,'Renal Disease'] <- x1[i,13]
  x[i,'Diabetes'] <- max(x1[i,10],x1[i,11],na.rm=T)
  x[i,'Nervous System Disease'] <- max(x1[i,4],x1[i,5],na.rm=T)
  x[i,'Malignant Tumor'] <- max(x1[i,14],x1[i,16],na.rm=T)
}


# data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv"); # EICU
# x = data[,83:90]


colname <- colnames(x)

for (i in c(1:length(colname))){
  colname[i] <- gsub('[.]', ' ', colname[i])
}
com_t <- data.frame(
  Comorbidity = colname,
  id = c(1:8)
)
for (i in c(1:8)){
  com_t[i,"A"] <-mean(x[data$kmeans==0,i],na.rm=T)
  com_t[i,"B"] <-mean(x[data$kmeans==1,i],na.rm=T)
  com_t[i,"C"] <-mean(x[data$kmeans==2,i],na.rm=T)
  com_t[i,"D"] <-mean(x[data$kmeans==3,i],na.rm=T)
}
com_t$id <- c(1:8)
AddRow=c(NA,nrow(com_t)+1, com_t[1,(ncol(com_t)-4):ncol(com_t)])
com_t=rbind(com_t,as.numeric(AddRow)) 
com_t$Comorbidity = factor(com_t$Comorbidity, levels = com_t$Comorbidity[order(-com_t$id)])

rmax <- ceil(max(com_t[1:8,3:6])*100)/100

coord_radar<-function(theta="x",start=0,direction=1)
{theta<-match.arg(theta,c("x","y"))
r<-if(theta=="x") 
  "y" 
else "x"
ggproto("CoordRadar",CoordPolar,theta=theta,r=r,start=start,
        direction=sign(direction),
        is_linear=function(coord) TRUE)}

p_A <- ggplot()+
  geom_polygon(data=com_t,aes(x=id,y=A),color="cyan",fill= "cyan",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(0,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=com_t[1:8,],aes(x=id,y=A,group = Comorbidity,color=Comorbidity),size=2.5)+ xlab("Phenotype A")+
  theme(legend.position="none") #隐藏图例

p_B <- ggplot()+
  geom_polygon(data=com_t,aes(x=id,y=B),color="maroon",fill= "maroon",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(0,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=com_t[1:8,],aes(x=id,y=B,group = Comorbidity,color=Comorbidity),size=2.5)+ xlab("Phenotype B")+
  theme(legend.position="none") #隐藏图例

p_C <- ggplot()+
  geom_polygon(data=com_t,aes(x=id,y=C),color="orange",fill= "orange",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(0,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=com_t[1:8,],aes(x=id,y=C,group = Comorbidity,color=Comorbidity),size=2.5)+ xlab("Phenotype C")+
  theme(legend.position="none") #隐藏图例

p_D <- ggplot()+
  geom_polygon(data=com_t,aes(x=id,y=D),color="wheat",fill= "wheat",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(0,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=com_t[1:8,],aes(x=id,y=D,group = Comorbidity,color=Comorbidity),size=2.5)+ xlab("Phenotype D")
p <- list()
p[[1]] <- p_A
p[[2]] <- p_B
p[[3]] <- p_C
p[[4]] <- p_D
f1 <- wrap_plots(p, ncol = 4)
ggsave(
  # filename = paste("./result/sub/合并症_eicu.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  filename = paste("./result/sub/合并症_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 14,             # 宽
  height = 3,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)
#-----Scales------#
data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv"); # EICU
x = data[,101:104]

# data <- read.csv("./data/xg2.csv"); # MIMIC
# x = data[,c(98:99,101,102)]

x <- scale(x)
colname <- colnames(x)
for (i in c(1:length(colname))){
  colname[i] <- gsub('[.]', ' ', colname[i])
}
scale_t <- data.frame(
  Scale = colname,
  id = c(1:length(colname))
)

for (i in c(1:length(colname))){
  scale_t[i,"A"] <-mean(x[data$kmeans==0,i],na.rm=T)
  scale_t[i,"B"] <-mean(x[data$kmeans==1,i],na.rm=T)
  scale_t[i,"C"] <-mean(x[data$kmeans==2,i],na.rm=T)
  scale_t[i,"D"] <-mean(x[data$kmeans==3,i],na.rm=T)
}
scale_t$id <- c(1:length(colname))
AddRow=c(NA,nrow(scale_t)+1, scale_t[1,3:ncol(scale_t)])
scale_t=rbind(scale_t,as.numeric(AddRow))
rmax <- ceil(max(scale_t[1:4,3:6])*100)/100
rmin <- -0.5
coord_radar<-function(theta="x",start=0,direction=1)
{theta<-match.arg(theta,c("x","y"))
r<-if(theta=="x") 
  "y" 
else "x"
ggproto("CoordRadar",CoordPolar,theta=theta,r=r,start=start,
        direction=sign(direction),
        is_linear=function(coord) TRUE)}

p_A <- ggplot()+
  geom_polygon(data=scale_t,aes(x=id,y=A),color="cyan",fill= "cyan",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(rmin,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=scale_t[1:length(colname),],aes(x=id,y=A,group = Scale,color=Scale),size=2.5)+ xlab("Phenotype A")+
  theme(legend.position="none") #隐藏图例

p_B <- ggplot()+
  geom_polygon(data=scale_t,aes(x=id,y=B),color="maroon",fill= "maroon",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(rmin,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=scale_t[1:length(colname),],aes(x=id,y=B,group = Scale,color=Scale),size=2.5)+ xlab("Phenotype B")+
  theme(legend.position="none") #隐藏图例

p_C <- ggplot()+
  geom_polygon(data=scale_t,aes(x=id,y=C),color="orange",fill= "orange",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(rmin,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=scale_t[1:length(colname),],aes(x=id,y=C,group = Scale,color=Scale),size=2.5)+ xlab("Phenotype C")+
  theme(legend.position="none") #隐藏图例

p_D <- ggplot()+
  geom_polygon(data=scale_t,aes(x=id,y=D),color="wheat",fill= "wheat",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(rmin,rmax)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=scale_t[1:length(colname),],aes(x=id,y=D,group = Scale,color=Scale),size=2.5)+ xlab("Phenotype D")
p <- list()
p[[1]] <- p_A
p[[2]] <- p_B
p[[3]] <- p_C
p[[4]] <- p_D
f1 <- wrap_plots(p, ncol = 4)

ggsave(
  filename = paste("./result/sub/评估量表_eicu.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  # filename = paste("./result/sub/评估量表_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 14,             # 宽
  height = 3,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

#-----vasoactive drugs------#
data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv"); # EICU
x = data[,c(106,108,110,112,114,116,118)]

# data <- read.csv("./data/xg2.csv"); # MIMIC
# x = data[,c(104,106,108,110,112,114,116)]
x[is.na(x)] <- 0
x <- scale(x)
colname <- colnames(x)
for (i in c(1:length(colname))){
  colname[i] <- gsub('[.]', ' ', colname[i])
}
drug_t <- data.frame(
  Drug = colname,
  id = c(1:length(colname))
)

for (i in c(1:length(colname))){
  drug_t[i,"A"] <-mean(x[data$kmeans==0,i])
  drug_t[i,"B"] <-mean(x[data$kmeans==1,i])
  drug_t[i,"C"] <-mean(x[data$kmeans==2,i])
  drug_t[i,"D"] <-mean(x[data$kmeans==3,i])
}
# drug_t <- drug_t %>% arrange(A)
drug_t$id <- c(1:length(colname))
AddRow=c(NA,nrow(drug_t)+1, drug_t[1,3:ncol(drug_t)])
drug_t=rbind(drug_t,as.numeric(AddRow))
rmax <- ceil(max(drug_t[1:7,3:6])*100)/100

coord_radar<-function(theta="x",start=0,direction=1)
{theta<-match.arg(theta,c("x","y"))
r<-if(theta=="x") 
  "y" 
else "x"
ggproto("CoordRadar",CoordPolar,theta=theta,r=r,start=start,
        direction=sign(direction),
        is_linear=function(coord) TRUE)}

p_A <- ggplot()+
  geom_polygon(data=drug_t,aes(x=id,y=A),color="cyan",fill= "cyan",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(-0.25,0.5)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=drug_t[1:length(colname),],aes(x=id,y=A,group = Drug,color=Drug),size=2.5)+ xlab("Phenotype A")+
  theme(legend.position="none") #隐藏图例

p_B <- ggplot()+
  geom_polygon(data=drug_t,aes(x=id,y=B),color="maroon",fill= "maroon",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(-0.25,0.5)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=drug_t[1:length(colname),],aes(x=id,y=B,group = Drug,color=Drug),size=2.5)+ xlab("Phenotype B")+
  theme(legend.position="none") #隐藏图例

p_C <- ggplot()+
  geom_polygon(data=drug_t,aes(x=id,y=C),color="orange",fill= "orange",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(-0.25,0.5)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=drug_t[1:length(colname),],aes(x=id,y=C,group = Drug,color=Drug),size=2.5)+ xlab("Phenotype C")+
  theme(legend.position="none") #隐藏图例

p_D <- ggplot()+
  geom_polygon(data=drug_t,aes(x=id,y=D),color="wheat",fill= "wheat",alpha=0.3)+
  coord_radar()+
  #scale_x_continuous(breaks=com_t$id,labels=com_t$X)+ #,labels=com_t$X
  ylim(-0.25,0.5)+ 
  theme_bw()+
  theme(axis.text.x=element_text(size=0.1,colour="grey"
                                 #,angle = angle
  ))+
  ylab(" ")+
  geom_point(data=drug_t[1:length(colname),],aes(x=id,y=D,group = Drug,color=Drug),size=2.5)+ xlab("Phenotype D")

p <- list()
p[[1]] <- p_A
p[[2]] <- p_B
p[[3]] <- p_C
p[[4]] <- p_D
f1 <- wrap_plots(p, ncol = 4)
ggsave(
  # filename = paste("./result/sub/血管活性药物_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  filename = paste("./result/sub/血管活性药物_eicu.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 14,             # 宽
  height = 3,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

###-----survival analysis-------###
# data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv"); # EICU
# data$los_hospital <- (data$hospitaldischargeoffset-data$hospitaladmitoffset)/1440
# data$hospital_expire_flag <- data$hosp_mort

data <- read.csv("./data/xg2.csv"); # MIMIC

data[data$kmeans==0,"Phenotype"] <- "A"
data[data$kmeans==1,"Phenotype"] <- "B"
data[data$kmeans==2,"Phenotype"] <- "C"
data[data$kmeans==3,"Phenotype"] <- "D"

for (i in c(1:length(data[,1]))){
  if (data[i,"los_hospital"]<= 28 && data[i,"hospital_expire_flag"]==1){
    data[i,"die28"] <- 1
  }
  else {
    data[i,"die28"] <- 0
  }
  if (data[i,"los_hospital"]>= 28){
    data[i,"time"] <- 28
  }
  else {
    data[i,"time"] <- data[i,"los_hospital"]
  }
  if (data[i,"los_hospital"]<= 90 && data[i,"hospital_expire_flag"]==1){
    data[i,"die90"] <- 1
  }
  else {
    data[i,"die90"] <- 0
  }
  if (data[i,"los_hospital"]>= 90){
    data[i,"time90"] <- 90
  }
  else {
    data[i,"time90"] <- data[i,"los_hospital"]
  }
}

fit28 <- survfit(Surv(time, die28) ~ Phenotype, data = data)
fit90 <- survfit(Surv(time90, die90) ~ Phenotype, data = data)

grid.draw.ggsurvplot <- function(x){
  survminer:::print.ggsurvplot(x, newpage = FALSE)
}
f1 <- ggsurvplot(fit28,data = data,
                 fun = "cumhaz", 
                 pval = TRUE, 
                 censor.shape="|", censor.size = 1.5,
                 conf.int = TRUE, # 可信区间
                 risk.table = TRUE,
                 risk.table.abs_pct = TRUE,
                 risk.table.height = 0.3,
                 break.time.by = 7,
                 xlim = c(0,28),
                 risk.table.y.text.col = T, # risk table文字注释颜色
                 risk.table.y.text = FALSE, # risk table显示条形而不是文字
                 tables.theme = theme_bw(),
                 palette = "lancet", # 支持ggsci配色，自定义颜色，brewer palettes中的配色，等
                 ggtheme = theme_bw() # 支持ggplot2及其扩展包的主题
)

ggsave(
  plot = f1,
  filename = paste("./result/sub/累积风险曲线28_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  # filename = paste("./result/sub/累积风险曲线28_eicu.pdf"),
  width = 10,             # 宽
  height = 7,            # 高
  units = "in",          # 单位
  dpi = 300
)
f1 <- ggsurvplot(fit90,data = data,
                 fun = "cumhaz", 
                 pval = TRUE, 
                 censor.shape="|", censor.size = 1.5,
                 conf.int = TRUE, # 可信区间
                 risk.table = TRUE,
                 risk.table.abs_pct = TRUE,
                 risk.table.height = 0.3,
                 break.time.by = 15,
                 xlim = c(0,90),
                 risk.table.y.text.col = T, # risk table文字注释颜色
                 risk.table.y.text = FALSE, # risk table显示条形而不是文字
                 tables.theme = theme_bw(),
                 palette = "lancet", # 支持ggsci配色，自定义颜色，brewer palettes中的配色，等
                 ggtheme = theme_bw() # 支持ggplot2及其扩展包的主题
)

ggsave(
  plot = f1,
  filename = paste("./result/sub/累积风险曲线90_mimic.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  # filename = paste("./result/sub/累积风险曲线90_eicu.pdf"),
  width = 10,             # 宽
  height = 7,            # 高
  units = "in",          # 单位
  dpi = 300
)

###-----Baseline comparison of sepsis subphenotypes-------###
# MIMIC
data <- read.csv("./data/xg2.csv");
colname <- colnames(data)
colname
x = data[,c(18:78,4,8,103:116,117:118,98:102,119,120,10,122)]
x1 = data[,80:96]
for (i in c(1:33177)){
  x[i,'Myocardial Infarct'] <- x1[i,1]
  x[i,'Congestive Heart Failure'] <- x1[i,2]
  x[i,'Chronic Pulmonary Disease'] <- x1[i,6]
  x[i,'Liver Disease'] <- max(x1[i,9],x1[i,15],na.rm=T)
  x[i,'Renal Disease'] <- x1[i,13]
  x[i,'Diabetes'] <- max(x1[i,10],x1[i,11],na.rm=T)
  x[i,'Nervous System Disease'] <- max(x1[i,4],x1[i,5],na.rm=T)
  x[i,'Malignant Tumor'] <- max(x1[i,14],x1[i,16],na.rm=T)
}
colname <- colnames(x)
colname
x <- x[,c(62,63,1:61,89:96,80,81,83,84,64:79)]
colnames(x) <- gsub('[.]', ' ', colnames(x))
x[,'Sex'] <- factor(x[,'gender'], levels=c("F", "M"))
x$Age <- x$admission_age
x$`Hospital Mortality` = data$hospital_expire_flag
x$`Acute Death` <- data$acute_sepsis
x$CCI <- data$cci
x$`Hospital length of stay` = data$los_hospital
x$`ICU length of stay` = data$los_icu
x$Phenotype <- data$kmeans
x[x$kmeans==0,"Phenotype"] = "A"
x[x$kmeans==1,"Phenotype"] = "B"
x[x$kmeans==2,"Phenotype"] = "C"
x[x$kmeans==3,"Phenotype"] = "D"
x$Phenotype <- factor(x$Phenotype)
vars <- c("Sex","Age","Mean HR","Minimum HR","Maximum HR","Mean SBP","Minimum SBP",
          "Maximum SBP","Mean DBP","Minimum DBP","Maximum DBP","Mean MAP","Minimum MAP",
          "Maximum MAP","Mean RR","Minimum RR","Maximum RR","Mean Temperature",
          "Minimum Temperature","Maximum Temperature","Mean PO2","Minimum PO2","Mean PCO2",
          "Minimum PCO2","Maximum PCO2","Mean FiO2","Minimum FiO2","Maximum FiO2",
          "Mean Base Excess","Minimum Base Excess","Maximum Base Excess","Mean RBC",
          "Maximum Hemoglobin","Mean RDW","Mean MCH","Mean MCV","Mean MCHC","Mean Platelet",
          "Mean WBC","Mean Basophils","Minimum Basophils","Maximum Basophils","Mean Eosinophils",
          "Minimum Eosinophils","Mean Lymphocytes","Mean Monocytes","Maximum Monocytes",
          "Minimum Neutrophils","Mean Albumin","Minimum Albumin","Maximum Albumin","Mean AG",
          "Mean BUN","Mean Calcium","Mean Chloride","Mean Sodium","Mean Potassium","Mean Glucose",
          "Minimum Glucose","Mean Creatinine","Mean INR","Mean PTT","Minimum PTT",
          'Myocardial Infarct','Congestive Heart Failure','Chronic Pulmonary Disease',
          'Liver Disease','Renal Disease','Diabetes','Nervous System Disease','Malignant Tumor',
          'SOFA','APSIII','OASIS','GCS',"Maximum Dopamine","Mean Dopamine","Maximum Epinephrine",
          "Mean Epinephrine","Maximum Norepinephrine","Mean Norepinephrine","Maximum Phenylephrine",
          "Mean Phenylephrine","Maximum Vasopressin","Mean Vasopressin","Maximum Dobutamine",
          "Mean Dobutamine","Maximum Milrinone","Mean Milrinone","CRRT","Ventilation",
          "Hospital Mortality","Acute Death","CCI","Hospital length of stay","ICU length of stay")
fvars<-c("Sex",'Myocardial Infarct','Congestive Heart Failure','Chronic Pulmonary Disease',
         'Liver Disease','Renal Disease','Diabetes','Nervous System Disease','Malignant Tumor',
         "CRRT","Ventilation","Hospital Mortality","Acute Death","CCI")
strata<-"Phenotype"
table_mimic<-scitb1(vars=vars,fvars=fvars,strata=strata,data=x,statistic=T,smd = F,Overall=T)
write.csv(table_mimic,'result/sub/Baseline_mimic.csv')


# EICU
data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv");
colname <- colnames(data)
colname
x = data[,c(16,14,22:82,83:90,101:104,105:120)]
colname <- colnames(x)
colname
for (i in c(64:71)){
  x[is.na(x[,i]),i] <- 0
}
colnames(x) <- gsub('[.]', ' ', colnames(x))
x[is.na(x[,'gender']),'gender'] <- 0
x[x[,'gender']==0,'Sex'] <- "F"
x[x[,'gender']==1,'Sex'] <- "M"
x[,'gender'] <- factor(x[,'gender'], levels=c("F", "M"))
x$Age <- x$age
x$`Hospital Mortality` = data$hosp_mort
x$`Acute Death` <- data$acute_sepsis
x$CCI <- data$cci
x$`Hospital length of stay` = (data$hospitaldischargeoffset-data$hospitaladmitoffset)/1440
x$`ICU length of stay` = data$icu_los_days
x$Phenotype <- data$kmeans
x[data$kmeans==0,"Phenotype"] = "A"
x[data$kmeans==1,"Phenotype"] = "B"
x[data$kmeans==2,"Phenotype"] = "C"
x[data$kmeans==3,"Phenotype"] = "D"
x$Phenotype <- factor(x$Phenotype)
vars <- c("Sex","Age","Mean HR","Minimum HR","Maximum HR","Mean SBP","Minimum SBP",
          "Maximum SBP","Mean DBP","Minimum DBP","Maximum DBP","Mean MAP","Minimum MAP",
          "Maximum MAP","Mean RR","Minimum RR","Maximum RR","Mean Temperature",
          "Minimum Temperature","Maximum Temperature","Mean PO2","Minimum PO2","Mean PCO2",
          "Minimum PCO2","Maximum PCO2","Mean FiO2","Minimum FiO2","Maximum FiO2",
          "Mean Base Excess","Minimum Base Excess","Maximum Base Excess","Mean RBC",
          "Maximum Hemoglobin","Mean RDW","Mean MCH","Mean MCV","Mean MCHC","Mean Platelet",
          "Mean WBC","Mean Basophils","Minimum Basophils","Maximum Basophils","Mean Eosinophils",
          "Minimum Eosinophils","Mean Lymphocytes","Mean Monocytes","Maximum Monocytes",
          "Minimum Neutrophils","Mean Albumin","Minimum Albumin","Maximum Albumin","Mean AG",
          "Mean BUN","Mean Calcium","Mean Chloride","Mean Sodium","Mean Potassium","Mean Glucose",
          "Minimum Glucose","Mean Creatinine","Mean INR","Mean PTT","Minimum PTT",
          'Myocardial Infarct','Congestive Heart Failure','Chronic Pulmonary Disease',
          'Liver Disease','Renal Disease','Diabetes','Nervous System Disease','Malignant Tumor',
          'SOFA','APSIII','OASIS','GCS',"Maximum Dopamine","Mean Dopamine","Maximum Epinephrine",
          "Mean Epinephrine","Maximum Norepinephrine","Mean Norepinephrine","Maximum Phenylephrine",
          "Mean Phenylephrine","Maximum Vasopressin","Mean Vasopressin","Maximum Dobutamine",
          "Mean Dobutamine","Maximum Milrinone","Mean Milrinone","CRRT","Ventilation",
          "Hospital Mortality","Acute Death","CCI","Hospital length of stay","ICU length of stay")
fvars<-c("Sex",'Myocardial Infarct','Congestive Heart Failure','Chronic Pulmonary Disease',
         'Liver Disease','Renal Disease','Diabetes','Nervous System Disease','Malignant Tumor',
         "CRRT","Ventilation","Hospital Mortality","Acute Death","CCI")
strata<-"Phenotype"
table_eicu<-scitb1(vars=vars,fvars=fvars,strata=strata,data=x,statistic=T,smd = F,Overall=T)
write.csv(table_eicu,'result/sub/Baseline_eicu.csv')

###-----Machine learning-based sepsis subphenotype prediction model-------###
# Note: See machine_learning.py for the specific training process of machine learning. 
# This part is only used to draw specific ROC curves and changes in AUC.
###-----Features screening of sepsis subphenotypes prediction model
f <- list()
n <- 1
path <- c('BPNN.csv','XGOOST.csv','SVM.csv','RF.csv')
name <- c('BPNN','XGBoost','SVM','RF')
for (i in c(1:4)){
  data_NN <- read.csv(paste0("./result/model/",path[i]))
  print(paste0("./result/model/",path[i]))
  data_mimic_NN <- data_NN[,1:4]
  data_eicu_NN <- data_NN[,5:8]
  colnames(data_mimic_NN) <- c('Phenotype A','Phenotype B','Phenotype C','Phenotype D')
  colnames(data_eicu_NN) <- c('Phenotype A','Phenotype B','Phenotype C','Phenotype D')
  data_mimic_NN$Mean <- apply(data_mimic_NN,1,mean)
  data_eicu_NN$Mean <- apply(data_eicu_NN,1,mean)
  data_mimic_NN$feature <- c(1:20)
  data_eicu_NN$feature <- c(1:20)
  data_mimic_NN <- data_mimic_NN[order(data_mimic_NN$feature,decreasing = T), ]
  data_eicu_NN <- data_eicu_NN[order(data_eicu_NN$feature,decreasing = T), ]
  
  
  mydata_mimic <- melt(data_mimic_NN,id="feature")
  colnames(mydata_mimic) <- c("feature","Phenotype","AUC")
  mydata_eicu <- melt(data_eicu_NN,id="feature")
  colnames(mydata_eicu) <- c("feature","Phenotype","AUC")
  
  f[[n]] <- ggplot(mydata_mimic,aes(x=feature,y=AUC,group=Phenotype,color=Phenotype,shape=Phenotype))+
    geom_line(size=1.2)+scale_y_continuous(limits = c(0.35, 1), breaks = seq(0.35, 1, by = 0.1))+scale_x_reverse()+
    theme_bw() +#去掉背景灰色
    xlab('Feature Number')+
    ggtitle('MIMIC')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 10,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA),
          # panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
          # text = element_text(family = "STXihei"),#设置中文字体的显示
          # legend.position = c(.075,.915),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color="black"))+#为图例田间边框线
    scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))+
    theme(legend.position="none")+ #隐藏图例
    ggplot(mydata_eicu,aes(x=feature,y=AUC,group=Phenotype,color=Phenotype,shape=Phenotype))+
    geom_line(size=1.2)+scale_y_continuous(limits = c(0.35, 1), breaks = seq(0.35, 1, by = 0.1))+scale_x_reverse()+
    ggtitle('eICU')+
    xlab('Feature Number')+
    theme_bw() +#去掉背景灰色
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 10,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA),
          # panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
          # text = element_text(family = "STXihei"),#设置中文字体的显示
          # legend.position = c(.075,.915),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color="black"))+#为图例田间边框线
    scale_color_manual(values=c("skyblue", "maroon", "orange", "gold", "brown1"))+
    labs(col = name[i])+
    theme(legend.title = element_text(size = 8,face = "bold",     #字体加粗
                                      color = "black",      #字体颜色
                                      hjust = 0.5,          #字体左右的位置
                                      vjust = 0.5))+ #颜色
    geom_vline(xintercept = 11, linetype = "dashed")+
    annotate("text", x = 11-2.8, y = 0.4, label = paste0("The best features = ",11),size = 2)
  
  n <- n+1
}
f1 <- wrap_plots(f, ncol = 1)
ggsave(
  filename = paste("./result/sub/ROC.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 9,             # 宽
  height = 12,            # 高
  units = "in",          # 单位
  dpi = 800              # 分辨率DPI
)

###----ROCt for sepsis subphenotype prediction optimal model-------###
y_mimic <- read.csv("./data/xg2.csv")[,c("kmeans")]
y_eicu <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv")[,c("kmeans")]
BPNN_mimic <- read.csv("./data/BPNN_subphenotype_mimic.csv")
BPNN_eicu <- read.csv("./data/BPNN_subphenotype_eicu.csv")
name <- c('Phenotype A','Phenotype B','Phenotype C','Phenotype D')
p <- list()
for (i in c(0:3)){
  BPNN_mimic$y <- ifelse(y_mimic == i, 1, 0)
  BPNN_mimic$x <- BPNN_mimic[,i+1]
  BPNN_eicu$y <- ifelse(y_eicu == i, 1, 0)
  BPNN_eicu$x <- BPNN_eicu[,i+1]
  roc1 <- pROC::roc(BPNN_mimic$y, BPNN_mimic$x,ci=TRUE)
  roc2 <- pROC::roc(BPNN_eicu$y, BPNN_eicu$x,ci=TRUE)
  lab1 <- paste0('AUC of MIMIC: ',round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
  lab2 <- paste0('AUC of eICU: ',round(roc2$ci[2],2),'(',round(roc2$ci[1],2),',',round(roc2$ci[3],2),')')
  p[[i+1]] <- ggroc(list(MIMIC=roc1, EICU=roc2),size = 1.2)+
    labs(col = name[i+1])+
    theme_bw()+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 10,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          legend.position = c(0.74,0.15),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c(lab1,lab2))
}
f1 <- wrap_plots(p, ncol = 2)
ggsave(
  filename = paste("./result/sub/BPNN_AUC.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 10,             # 宽
  height = 10,            # 高
  units = "in",          # 单位
  dpi = 600              # 分辨率DPI
)
###-----SHAP plot for sepsis subphenotype prediction optimal model-------###
# Note: See SHAP_sepsis_subphenotype.py for the specific training process of sepsis subphenotype prediction optimal mode's SHAP plot. 
A <- read.csv("./result/model/shap_f0.csv")
B <- read.csv("./result/model/shap_f1.csv")
C <- read.csv("./result/model/shap_f2.csv")
D <- read.csv("./result/model/shap_f3.csv")

function_phenotype <- function(death,yname,color = "wheat"){
  a <- data.frame(
    name = c("Mean PO2","Maximum HR","Mean HR","Mean MCHC","Mean RR","Maximum FiO2","Mean RDW","Maximum DBP","Mean BUN","Maximum Hemoglobin","Maximum SBP")
  )
  for (i in c(1:11)){
    a[i,'mean'] <- mean(death[,i])
    a[i,'sd'] <- sd(death[,i])
    a[i,'cv'] <- a[i,'mean']/a[i,'sd']
    a[i,'down'] <- t.test(death[,i])$conf.int[1]
    a[i,'up'] <- t.test(death[,i])$conf.int[2]
  }
  sorted_df <- a[order(a$mean,decreasing = T), ]
  sorted_df$name = factor(sorted_df$name, levels = sorted_df$name[order(sorted_df$mean)])
  f <- ggplot(data = sorted_df[1:11,], aes(x = name, y = mean)) +
    geom_bar(stat = "identity", 
             width = 0.8, colour = color, size = 0.25,
             fill = color, alpha = 1) +
    # ylim(0, 0.04) + # 设置y轴范围
    theme(
      axis.title = element_text(size = 10, face = "plain", color = "black"), # 设置标题的字体及大小
      axis.text = element_text(size = 8, face = "plain", color = "black") # 设置坐标轴的字体及大小
    )+
    coord_flip()+
    xlab(" ")+
    ylab(yname)+
    geom_errorbar(data = sorted_df[1:11,], aes(x = name, 
                                               ymin = down, 
                                               ymax = up, 
                                               width = 0.8), 
                  position = position_dodge(width = 5))
  return(list(a,f))
}

F1 <- function_phenotype(A,'Mean(|SHAP value|) of phenotype A','cyan')
a1 <- F1[[1]]
f1 <- F1[[2]]
F2 <- function_phenotype(B,'Mean(|SHAP value|) of phenotype B','maroon')
a2 <- F2[[1]]
f2 <- F2[[2]]
F3 <- function_phenotype(C,'Mean(|SHAP value|) of phenotype C','orange')
a3 <- F3[[1]]
f3 <- F3[[2]]
F4 <- function_phenotype(D,'Mean(|SHAP value|) of phenotype D','wheat')
a4 <- F4[[1]]
f4 <- F4[[2]]


f <- cowplot::plot_grid(f1, f2,f3,f4,ncol = 4)
ggsave(
  filename = paste("./result/sub/SHAP_phenotype.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f,
  width = 20,             # 宽
  height = 4,            # 高
  units = "in",          # 单位
  dpi = 800              # 分辨率DPI
)