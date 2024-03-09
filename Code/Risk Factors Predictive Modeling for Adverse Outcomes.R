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

###-----61 base features to establish a preliminary prediction model for adverse outcome-------###

y_mimic <- read.csv("./data/xg2.csv")[,c("hospital_expire_flag","acute_sepsis","cci")]
y_eicu <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv")[,c("hosp_mort","acute_sepsis","cci")]

BPNN_mimic <- read.csv("./result/model/BPNN_mimic_all.csv")
BPNN_eicu <- read.csv("./result/model/BPNN_eicu_all.csv")
XGBOOST_mimic <- read.csv("./result/model/XGBOOST_mimic_all.csv")
XGBOOST_eicu <- read.csv("./result/model/XGBOOST_eicu_all.csv")
SVM_mimic <- read.csv("./result/model/SVM_mimic_all.csv")
SVM_eicu <- read.csv("./result/model/SVM_eicu_all.csv")
RF_mimic <- read.csv("./result/model/RF_mimic_all.csv")
RF_eicu <- read.csv("./result/model/RF_eicu_all.csv")


name <- c('Hospital Mortality','Acute Death','CCI')
f <- list()
t_mimic <- data.frame(
  matrix(c(1:16),nrow = 4,ncol = 4)
)
colnames(t_mimic) <- c('cutoff','sensitivity','specificity','AUC(CI)')
rownames(t_mimic) <- c('BPNN','XGBoost','SVM','RF')
t_eicu <- data.frame(
  matrix(c(1:16),nrow = 4,ncol = 4)
)
colnames(t_eicu) <- c('cutoff','sensitivity','specificity','AUC(CI)')
rownames(t_eicu) <- c('BPNN','XGBoost','SVM','RF')
for (i in c(1:3)){
  roc1 <- pROC::roc(y_mimic[,i], BPNN_mimic[,i],ci=TRUE)
  roc2 <- pROC::roc(y_mimic[,i], XGBOOST_mimic[,i],ci=TRUE)
  roc3 <- pROC::roc(y_mimic[,i], SVM_mimic[,i],ci=TRUE)
  roc4 <- pROC::roc(y_mimic[,i], RF_mimic[,i],ci=TRUE)
  
  roc5 <- pROC::roc(y_eicu[,i], BPNN_eicu[,i],ci=TRUE)
  roc6 <- pROC::roc(y_eicu[,i], XGBOOST_eicu[,i],ci=TRUE)
  roc7 <- pROC::roc(y_eicu[,i], SVM_eicu[,i],ci=TRUE)
  roc8 <- pROC::roc(y_eicu[,i], RF_eicu[,i],ci=TRUE)
  
  BPNN_mimic1 <- cutoff::roc( BPNN_mimic[,i],y_mimic[,i])
  XGBOOST_mimic1 <- cutoff::roc( XGBOOST_mimic[,i],y_mimic[,i])
  SVM_mimic1 <- cutoff::roc( SVM_mimic[,i],y_mimic[,i])
  RF_mimic1 <- cutoff::roc( RF_mimic[,i],y_mimic[,i])
  t_mimic[1,1:3] <- BPNN_mimic1[1,3:5]
  t_mimic[2,1:3] <- XGBOOST_mimic1[1,3:5]
  t_mimic[3,1:3] <- SVM_mimic1[1,3:5]
  t_mimic[4,1:3] <- RF_mimic1[1,3:5]
  t_mimic[1,4] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
  t_mimic[2,4] <- paste0(round(roc2$ci[2],2),'(',round(roc2$ci[1],2),',',round(roc2$ci[3],2),')')
  t_mimic[3,4] <- paste0(round(roc3$ci[2],2),'(',round(roc3$ci[1],2),',',round(roc3$ci[3],2),')')
  t_mimic[4,4] <- paste0(round(roc4$ci[2],2),'(',round(roc4$ci[1],2),',',round(roc4$ci[3],2),')')
  
  
  BPNN_eicu1 <- cutoff::roc( BPNN_eicu[,i],y_eicu[,i])
  XGBOOST_eicu1 <- cutoff::roc( XGBOOST_eicu[,i],y_eicu[,i])
  SVM_eicu1 <- cutoff::roc( SVM_eicu[,i],y_eicu[,i])
  RF_eicu1 <- cutoff::roc( RF_eicu[,i],y_eicu[,i])
  t_eicu[1,1:3] <- BPNN_eicu1[1,3:5]
  t_eicu[2,1:3] <- XGBOOST_eicu1[1,3:5]
  t_eicu[3,1:3] <- SVM_eicu1[1,3:5]
  t_eicu[4,1:3] <- RF_eicu1[1,3:5]
  t_eicu[1,4] <- paste0(round(roc5$ci[2],2),'(',round(roc5$ci[1],2),',',round(roc5$ci[3],2),')')
  t_eicu[2,4] <- paste0(round(roc6$ci[2],2),'(',round(roc6$ci[1],2),',',round(roc6$ci[3],2),')')
  t_eicu[3,4] <- paste0(round(roc7$ci[2],2),'(',round(roc7$ci[1],2),',',round(roc7$ci[3],2),')')
  t_eicu[4,4] <- paste0(round(roc8$ci[2],2),'(',round(roc8$ci[1],2),',',round(roc8$ci[3],2),')')
  
  g1 <- ggroc(list(BPNN=roc1, XGBoost=roc2,SVM=roc3,RF=roc4),size = 1.2)+ ggtitle('MIMIC')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 10,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          legend.position = c(.8,.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color="black"))+
    theme(legend.position="none")
  g2 <- ggroc(list(BPNN=roc5, XGBoost=roc6,SVM=roc7,RF=roc8),size = 1.2)+ ggtitle('eICU')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 10,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.background = element_rect(fill = "transparent",colour = NA),
          # plot.background = element_rect(fill = "transparent",colour = NA),
          # panel.grid.minor = element_blank(),#以上theme中代码用于去除网格线且保留坐标轴边框
          # text = element_text(family = "STXihei"),#设置中文字体的显示
          # legend.position = c(.8,.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color="black"))+
    labs(col = name[i])+
    theme(legend.title = element_text(size = 8,face = "bold",     #字体加粗
                                      color = "black",      #字体颜色
                                      hjust = 0.5,          #字体左右的位置
                                      vjust = 0.5))
  g3 <- ggtexttable(t_mimic,theme = ttheme("light",base_size = 6))
  g4 <- ggtexttable(t_eicu,theme = ttheme("light",base_size = 6))
  lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(7, 9))
  lay34 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(7, 7,2))
  lay1234 <- lay_bind_row(lay12, lay34, heights = c(4, 1.5))
  lay_show(lay1234)
  plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
  f[[i]] <- lay_grid(plots4, lay1234)
}
f1 <- wrap_plots(f, ncol = 1)
ggsave(
  filename = paste("./result/sub/ROC判断全部.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 9,             # 宽
  height = 12.5,            # 高
  units = "in",          # 单位
  dpi = 800              # 分辨率DPI
)

###-----SHAP plot for the preliminary prediction model for adverse outcome-------###
death <- read.csv("./result/model/shap_all0.csv")
acu <- read.csv("./result/model/shap_all1.csv")
cci <- read.csv("./result/model/shap_all2.csv")

function3 <- function(death,yname,color = "wheat"){
  a <- data.frame(
    name = gsub('[.]', ' ', colnames(death))
  )
  for (i in c(1:61)){
    a[i,'mean'] <- mean(death[,i])
    a[i,'sd'] <- sd(death[,i])
    a[i,'cv'] <- a[i,'mean']/a[i,'sd']
    a[i,'down'] <- t.test(death[,i])$conf.int[1]
    a[i,'up'] <- t.test(death[,i])$conf.int[2]
  }
  sorted_df <- a[order(a$mean,decreasing = T), ]
  sorted_df$name = factor(sorted_df$name, levels = sorted_df$name[order(sorted_df$mean)])
  f <- ggplot(data = sorted_df[1:20,], aes(x = name, y = mean)) +
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
    geom_errorbar(data = sorted_df[1:20,], aes(x = name, 
                                               ymin = down, 
                                               ymax = up, 
                                               width = 0.8), 
                  position = position_dodge(width = 5))
  return(list(a,f))
}

F1 <- function3(death,'Mean(|SHAP value|) of Hospital Mortality')
a1 <- F1[[1]]
f1 <- F1[[2]]
F2 <- function3(acu,'Mean(|SHAP value|) of Acute Death','salmon')
a2 <- F2[[1]]
f2 <- F2[[2]]
F3 <- function3(cci,'Mean(|SHAP value|) of CCI','orange')
a3 <- F3[[1]]
f3 <- F3[[2]]
f <- cowplot::plot_grid(f1, f2,f3,ncol = 3)
ggsave(
  filename = paste("./result/sub/SHAP前20.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f,
  width = 15,             # 宽
  height = 4,            # 高
  units = "in",          # 单位
  dpi = 800              # 分辨率DPI
)

###-----Features screening of base features-------###
f <- list()
n <- 1
path <- c('BPNN_all.csv','XGOOST_all.csv','SVM_all.csv','RF_all.csv')
name1 <- c('BPNN','XGBoost','SVM','RF')
name <- c('Hospital Mortality','Acute Death','CCI')
xx <- c(13,16,14)
for (i in c(1:3)){
  data_mimic_NN <- data.frame(
    feature <- c(1:20)
  )
  data_eicu_NN <- data.frame(
    feature <- c(1:20)
  )
  for (j in c(1:4)){
    data_NN <- read.csv(paste0("./result/model/",path[j]))
    data_mimic_NN[,name1[j]] <- data_NN[,i]
    data_eicu_NN[,name1[j]] <- data_NN[,i+3]
  }
  colnames(data_mimic_NN) <- c('feature','BPNN','XGBoost','SVM','RF')
  colnames(data_eicu_NN) <- c('feature','BPNN','XGBoost','SVM','RF')
  
  data_mimic_NN <- data_mimic_NN[order(data_mimic_NN$feature,decreasing = T), ]
  data_eicu_NN <- data_eicu_NN[order(data_eicu_NN$feature,decreasing = T), ]
  
  
  mydata_mimic <- melt(data_mimic_NN,id="feature")
  colnames(mydata_mimic) <- c("feature","ML","AUC")
  mydata_eicu <- melt(data_eicu_NN,id="feature")
  colnames(mydata_eicu) <- c("feature","ML","AUC")
  
  f[[n]] <- ggplot(mydata_mimic,aes(x=feature,y=AUC,group=ML,color=ML,shape=ML))+
    geom_line(size=1.2)+scale_y_continuous(limits = c(0.4, 0.9), breaks = seq(0.35, 1, by = 0.1))+scale_x_reverse()+
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
    ggplot(mydata_eicu,aes(x=feature,y=AUC,group=ML,color=ML,shape=ML))+
    geom_line(size=1.2)+scale_y_continuous(limits = c(0.4, 0.9), breaks = seq(0.35, 1, by = 0.1))+scale_x_reverse()+
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
    geom_vline(xintercept = xx[i], linetype = "dashed")+
    annotate("text", x = xx[i]-3, y = 0.4, label = paste0("The best features = ",xx[i]),size = 2)
  
  n <- n+1
}
f1 <- wrap_plots(f, ncol = 1)
ggsave(
  filename = paste("./result/sub/ROC_all.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 9,             # 宽
  height = 12,            # 高
  units = "in",          # 单位
  dpi = 800              # 分辨率DPI
)

###-----Risk factors prediction model && Forest plot-------###
#Note: This font library is only used in the following code, in other code please remove the library to prevent the image font anomaly!
library(showtext)
font_add('Arial','/C:/Windows/Fonts/Arial.ttf') #加载字体，MAC 中字体库在 /Library/Fonts
showtext_auto() #自动调用showtext，否则无法在ggsave()中使用，因为ggsave会自动打开和关闭图形设备。

# functions part
# Building formulas for R
Formula2 <- function(xxx,y = "hospital_expire_flag ~ ",n=8){ 
  l <- sapply(xxx, function(x) is.factor(x))
  ll <- c()
  for (z in c(1:(length(xxx)-n))){
    if (l[z]){
      if (all(xxx[,z] == xxx[1,z])){ll[z] <- F}
      else{ll[z] <- T}
    }
    else{ll[z] <- T}
  }
  for (z in c((length(xxx)-n+1):length(xxx))){ll[z] <- T}
  x3 <- xxx[, ll]
  xnam <- colnames(x3)[1:(length(colnames(x3))-n)]
  fmla <- as.formula(paste(y, paste0("`",xnam,"`", collapse= " + ")))
  return(list(x3,fmla))
}

# Drawing forest plots
or_plot2 <- function (logreg4=logreg4,title_key=1, table_text_size = 4, 
                      title_text_size = 13, plot_opts = NULL, table_opts = NULL, 
                      ...) 
{
  df.out <- data.frame()
  col1 <- colnames(logreg4)
  num0 <- strsplit(as.character(col1[3]),split='[()]')[[1]][2]
  num1 <- strsplit(as.character(col1[4]),split='[()]')[[1]][2]
  n <- 1
  for (i in c(1:length(logreg4[,1]))){
    n2 <- 0
    if(logreg4[i,2]=="Mean ± SD" && logreg4[i,6]!=""){
      split_strings <- strsplit(logreg4[i,6],split='[()]')[[1]]
      split_strings1 <- strsplit(as.character(split_strings[2]),split='[-]')[[1]]
      split_strings2 <- strsplit(as.character(split_strings1[2]),split='[,]')[[1]]
      OR <- as.numeric(split_strings[1])
      OR_L <- as.numeric(split_strings1[1])
      OR_U <- as.numeric(split_strings2[1])
      P <- paste0(substr(split_strings2[2], 1, 3), "0", substr(split_strings2[2], 4, nchar(split_strings2[2])))
      if (as.numeric(paste0("0", substr(split_strings2[2], 4, nchar(split_strings2[2]))))<=1)
      {
        df.out[n,"Name"] <- gsub("_", " ", logreg4[i,1])
        df.out[n,"Description"] <- logreg4[i,2]
        df.out[n,col1[3]] <- logreg4[i,3]
        df.out[n,col1[4]] <- logreg4[i,4]
        df.out[n,"OR(95%Cl,P)"] <- paste(OR,"(",as.numeric(OR_L),"-",as.numeric(OR_U),",",P,")", sep = "", collapse = "")
        df.out[n,"OR"] <- OR
        df.out[n,"OR_L"] <- OR_L
        df.out[n,"OR_U"] <- OR_U
        df.out[n,"P"] <- P
        df.out[n,"item"] <- logreg4[i,1]
        df.out[n,"idx"] <- n
        n2 <- 1
      }
      
    }
    if(logreg4[i,2]!="Mean ± SD" && logreg4[i,6]!=""){
      split_strings <- strsplit(logreg4[i,6],split='[()]')[[1]]
      split_strings1 <- strsplit(as.character(split_strings[2]),split='[-]')[[1]]
      split_strings2 <- strsplit(as.character(split_strings1[2]),split='[,]')[[1]]
      OR <- as.numeric(split_strings[1])
      OR_L <- as.numeric(split_strings1[1])
      OR_U <- as.numeric(split_strings2[1])
      P <- paste0(substr(split_strings2[2], 1, 3), "0", substr(split_strings2[2], 4, nchar(split_strings2[2])))
      if (as.numeric(paste0("0", substr(split_strings2[2], 4, nchar(split_strings2[2]))))<=1)
      {
        df.out[n,"Name"] <- gsub("_", " ", logreg4[i-1,1])
        df.out[n,"Description"] <- logreg4[i-1,2]
        df.out[n,col1[3]] <- logreg4[i-1,3]
        df.out[n,col1[4]] <- logreg4[i-1,4]
        df.out[n,"OR(95%Cl,P)"] <- ""
        df.out[n,"OR"] <- 1
        df.out[n,"OR_L"] <- NA
        df.out[n,"OR_U"] <- NA
        df.out[n,"P"] <- NA
        df.out[n,"item"] <- paste0(logreg4[i-1,1], logreg4[i-1,2])
        df.out[n,"idx"] <- n
        
        df.out[n+1,"Name"] <- ""
        df.out[n+1,"Description"] <- logreg4[i,2]
        df.out[n+1,col1[3]] <- logreg4[i,3]
        df.out[n+1,col1[4]] <- logreg4[i,4]
        df.out[n+1,"OR(95%Cl,P)"] <- paste(OR,"(",as.numeric(OR_L),"-",as.numeric(OR_U),",",P,")", sep = "", collapse = "")
        df.out[n+1,"OR"] <- OR
        df.out[n+1,"OR_L"] <- OR_L
        df.out[n+1,"OR_U"] <- OR_U
        df.out[n+1,"P"] <- P
        df.out[n+1,"item"] <- paste0(logreg4[i-1,1], logreg4[i,2])
        df.out[n+1,"idx"] <- n+1
        n2 <- 2
      }
    }
    n <- n+n2
  }
  df.out$item = factor(df.out$item, levels = df.out$item[order(-df.out$idx)])
  
  g1 = ggplot(df.out, aes(x = as.numeric(OR), xmin = as.numeric(OR_L), 
                          xmax = as.numeric(OR_U), y = item)) + geom_errorbarh(height = 0.2) + 
    geom_vline(xintercept = 1, linetype = "longdash", colour = "black",xlim = c(0,2),
               ticks_at = c(0,0.5,1,1.5,2)) + 
    geom_point(aes(size = Total), shape = 22, fill = "darkblue",size=4) + 
    scale_x_continuous(trans = "log10", breaks = scales::pretty_breaks()) + 
    xlab("Odds ratio (95% CI, log scale)") + theme_classic(11) + 
    theme(axis.title.x = element_text(), axis.title.y = element_blank(), 
          axis.text.y = element_blank(), axis.line.y = element_blank(), 
          axis.ticks.y = element_blank(), legend.position = "none")
  t1 = ggplot(df.out, aes(x = idx, y = item)) + 
    annotate("text", x = -1, y = df.out$item, 
             label = df.out$Name, hjust = 0, size = table_text_size) +
    annotate("text", x = -0.5, y = df.out$item, 
             label = df.out[, 2], hjust = 0, size = table_text_size) + 
    annotate("text", x = -0.15, y = df.out$item, 
             label = df.out[, 3], hjust = 0, size = table_text_size) + 
    annotate("text", x = 0.2, y = df.out$item, 
             label = df.out[, 4], hjust = 0, size = table_text_size) + 
    annotate("text", x = 1, y = df.out$item, 
             label = df.out[, 5], hjust = 1, size = table_text_size) + 
    theme_classic(14) + theme(axis.title.x = element_text(colour = "white"), 
                              axis.text.x = element_text(colour = "white"), axis.title.y = element_blank(), 
                              axis.text.y = element_blank(), axis.ticks.y = element_blank(), 
                              line = element_blank())
  
  g1 = g1 + plot_opts
  t1 = t1 + table_opts
  title = c(paste0("             Name                                   Description            Survival(",num0,")           Death(",num1,")                    OR(95%Cl,P value)"),
            paste0("             Name                                   Description            Not Death(",num0,")          Acute Death(",num1,")              OR(95%Cl,P value)"),
            paste0("             Name                                   Description            Not CCI(",num0,")            CCI(",num1,")                      OR(95%Cl,P value)"))
  p1 <- gridExtra::grid.arrange(t1, g1, ncol = 2, widths = c(3, 
                                                             2), top = grid::textGrob(title[title_key], x = 0.02, y = 0.2, gp = grid::gpar(fontsize = 10), 
                                                                                      just = "left"))
}

# building Risk factors prediction model
data <- read.csv("./data/xg2.csv")
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
for (i in c(1:61,63,64:77,80:84)){
  if (length(unique(x[,i]))>=10){
    mean_value <- mean(x[,i],na.rm=T)
    sd_value <- sd(x[,i],na.rm=T)
    x[is.na(x[,i]),i] <- mean_value
    x[x[,i]>(mean_value+20*sd_value),i] <- (mean_value+20*sd_value)
    x[x[,i]<(mean_value-20*sd_value),i] <- (mean_value-20*sd_value)
    
  }
}
Mean_Sd <- matrix(0, 2, 105)
for (i in c(1:61,63,64:77,80:84)){
  Mean_Sd[1,i] <- mean(x[,i],na.rm=T)
  Mean_Sd[2,i] <- sd(x[,i],na.rm=T)
  x[,i] <- (x[,i]-Mean_Sd[1,i])/Mean_Sd[2,i]
}
x[x[,62]==1,62] <- "M"
x[x[,62]==0,62] <- "F"
x[,62] <- factor(x[,62], levels=c("F", "M"))
for (i in c(89:96,78,79)){
  x[x[,i]==0,i] <- "No"
  x[x[,i]==1,i] <- "Yes"
  x[,i] <- factor(x[,i], levels=c("No", "Yes"))
}
colnames(x)[62] <- "Sex"
colnames(x)[63] <- "Age"
colnames(x) <-gsub('[.]', ' ', colnames(x))
summary(x)

data_eicu <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv");
x_eicu = data_eicu[,c(22:82,16,14,83:90,105:118,119:120,101:104,121,122,15,123)]
for (i in c(1:61,63,76:85,88:91)){
  if (length(unique(x_eicu[,i]))>=10){
    mean_value <- mean(x_eicu[,i],na.rm=T)
    sd_value <- sd(x_eicu[,i],na.rm=T)
    x_eicu[is.na(x_eicu[,i]),i] <- mean_value
    x_eicu[x_eicu[,i]>(mean_value+20*sd_value),i] <- (mean_value+20*sd_value)
    x_eicu[x_eicu[,i]<(mean_value-20*sd_value),i] <- (mean_value-20*sd_value)
    
  }
}
Mean_Sd <- matrix(0, 2, 105)
for (i in c(1:61,63,76:85,88:91)){
  Mean_Sd[1,i] <- mean(x_eicu[,i],na.rm=T)
  Mean_Sd[2,i] <- sd(x_eicu[,i],na.rm=T)
  x_eicu[,i] <- (x_eicu[,i]-Mean_Sd[1,i])/Mean_Sd[2,i]
}
for (i in c(62,64:71,86,87)){
  x_eicu[is.na(x_eicu[,i]),i] <- 0
}
x_eicu[x_eicu[,62]==1,62] <- "M"
x_eicu[x_eicu[,62]==0,62] <- "F"
x_eicu[,62] <- factor(x_eicu[,62], levels=c("F", "M"))
for (i in c(64:71,86,87)){
  x_eicu[x_eicu[,i]==0,i] <- "No"
  x_eicu[x_eicu[,i]==1,i] <- "Yes"
  x_eicu[,i] <- factor(x_eicu[,i], levels=c("No", "Yes"))
}
colnames(x_eicu)[94] <- 'hospital_expire_flag'
colnames(x_eicu)[62] <- "Sex"
colnames(x_eicu)[63] <- "Age"
colnames(x_eicu) <-gsub('[.]', ' ', colnames(x_eicu))

L <- list()
L[[1]] <- c("Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP")
L[[2]] <- c("Sex","Age")
L[[3]]<- c("Myocardial Infarct","Congestive Heart Failure","Chronic Pulmonary Disease",
           "Liver Disease","Renal Disease","Diabetes","Nervous System Disease","Malignant Tumor")
L[[4]] <- c("Maximum Dopamine","Mean Dopamine","Maximum Epinephrine",
            "Mean Epinephrine","Maximum Norepinephrine","Mean Norepinephrine","Maximum Phenylephrine",
            "Mean Phenylephrine","Maximum Vasopressin","Mean Vasopressin","Maximum Dobutamine",
            "Mean Dobutamine","Maximum Milrinone","Mean Milrinone","CRRT","Ventilation")
L[[5]] <- c("hospital_expire_flag")

# 对住院死亡率影响
R <- list()
result <- data.frame(
  model <- c('model-1','model-2','model-3','model-4'),
  name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
)
p_mimic <- list()
p_eicu <- list()
GLM <- list()
for (i in c(1:4)){
  lx <- c()
  for (j in c(1:i)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[,lxy]
  r <- Formula2(xxx,n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  GLM[[i]] <- logfit
  # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  pre <- predict(logfit,type='response')
  roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
  p_mimic[[i]] <- roc1
  roc11 <- cutoff::roc( pre,x3[,L[[5]]])
  result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
  xx_eciu <- x_eicu[,lx]
  pre_eicu <- predict(logfit,xx_eciu,type='response')
  roc1_eciu <- pROC::roc(x_eicu[,c(L[[5]])], pre_eicu,ci=TRUE)
  p_eicu[[i]] <- roc1_eciu
  roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[,c(L[[5]])])
  result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
  if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
  if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
  result[i,'P-value'] <- pp
  result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
}
colnames(result) <- c('Hospital Mortality','Features','AUC(MIMIC)','AUC(eICU)','P value')
R[[1]] <- result
g1 <- ggroc(p_mimic,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('MIMIC')+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('base','tj','com','treat'))+
  theme(legend.position="none")+
  theme(text = element_text(size=6))
g2 <- ggroc(p_eicu,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('eICU')+
  labs(col = 'Hospital Mortality',size=6)+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
  theme(text = element_text(size=6))
g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
g4 <- ggtexttable(R[[1]],rows = NULL,theme = ttheme("light",base_size = 5))
lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
lay_show(lay123)
plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
pdf(file = "./result/sub/LR_die_all.pdf",
    width = 5,             # 宽
    height = 3            # 高
)
f1 <- lay_grid(plots4, lay123)
dev.off()

PP <- c('Phenotype A','Phenotype B','Phenotype C','Phenotype D')
for (lab in c(0:3)){
  result <- data.frame(
    model <- c('model-1','model-2','model-3','model-4'),
    name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
  )
  p_mimic <- list()
  p_eicu <- list()
  GLM <- list()
  for (i in c(1:4)){
    lx <- c()
    for (j in c(1:i)){
      if (j>=2){
        cr <- c()
        for (z in c(1:length(L[[j]]))){
          f <- summary(glm(as.formula(paste0(L[[5]]," ~ ","`", L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
          if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
        }
        lx <- c(lx,L[[j]][cr])
      }else{lx <- c(lx,L[[1]])}
    }
    lxy <- c(lx,L[[5]])
    xxx <- x[data$kmeans==lab,lxy]
    r <- Formula2(xxx,n=1)
    x3 <- r[[1]]
    fmla <- r[[2]]
    logfit <- glm(fmla,data=x3,family="binomial")
    GLM[[i]] <- logfit
    # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
    pre <- predict(logfit,type='response')
    roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
    p_mimic[[i]] <- roc1
    roc11 <- cutoff::roc( pre,x3[,L[[5]]])
    result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
    xx_eciu <- x_eicu[data_eicu$kmeans==lab,lx]
    pre_eicu <- predict(logfit,xx_eciu,type='response')
    roc1_eciu <- pROC::roc(x_eicu[data_eicu$kmeans==lab,c(L[[5]])], pre_eicu,ci=TRUE)
    p_eicu[[i]] <- roc1_eciu
    roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[data_eicu$kmeans==lab,c(L[[5]])])
    result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
    if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
    if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
    result[i,'P-value'] <- pp
    result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
  }
  colnames(result) <- c('Hospital Mortality','Features','AUC(MIMIC)','AUC(eICU)','P value')
  R[[lab+2]] <- result
  g1 <- ggroc(p_mimic,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('MIMIC')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('base','tj','com','treat'))+
    theme(legend.position="none")+
    theme(text = element_text(size=6))
  g2 <- ggroc(p_eicu,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('eICU')+
    labs(col = PP[lab+1],size=6)+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
    theme(text = element_text(size=6))
  g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
  g4 <- ggtexttable(R[[lab+2]],rows = NULL,theme = ttheme("light",base_size = 5))
  lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
  lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
  lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
  lay_show(lay123)
  plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
  pdf(file = paste0("./result/sub/LR_die_",PP[lab+1],".pdf"),
      width = 5,             # 宽
      height = 3            # 高
  )
  f1 <- lay_grid(plots4, lay123)
  dev.off()
}
# 对急性死亡死亡率影响
L[[1]] <- c("Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose")
L[[5]] <- c('acute_sepsis')
R_acu <- list()
result <- data.frame(
  model <- c('model-1','model-2','model-3','model-4'),
  name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
)
p_mimic <- list()
p_eicu <- list()
GLM <- list()
for (i in c(1:4)){
  lx <- c()
  for (j in c(1:i)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[,lxy]
  r <- Formula2(xxx,y='acute_sepsis ~',n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  GLM[[i]] <- logfit
  # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  pre <- predict(logfit,type='response')
  roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
  p_mimic[[i]] <- roc1
  roc11 <- cutoff::roc( pre,x3[,L[[5]]])
  result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
  xx_eciu <- x_eicu[,lx]
  pre_eicu <- predict(logfit,xx_eciu,type='response')
  roc1_eciu <- pROC::roc(x_eicu[,c(L[[5]])], pre_eicu,ci=TRUE)
  p_eicu[[i]] <- roc1_eciu
  roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[,c(L[[5]])])
  result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
  if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
  if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
  result[i,'P-value'] <- pp
  result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
}
colnames(result) <- c('Acute Death','Features','AUC(MIMIC)','AUC(eICU)','P value')
R_acu[[1]] <- result
g1 <- ggroc(p_mimic,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('MIMIC')+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('base','tj','com','treat'))+
  theme(legend.position="none")+
  theme(text = element_text(family = "Times New Roman",
                            size=6))
g2 <- ggroc(p_eicu,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('eICU')+
  labs(col = 'Acute Death',size=6)+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
  theme(text = element_text(size=6))
g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
g4 <- ggtexttable(R_acu[[1]],rows = NULL,theme = ttheme("light",base_size = 5))
lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
lay_show(lay123)
plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
pdf(file = "./result/sub/LR_acu_all.pdf",
    width = 5,             # 宽
    height = 3            # 高
)
f1 <- lay_grid(plots4, lay123)
dev.off()

for (lab in c(0:3)){
  result <- data.frame(
    model <- c('model-1','model-2','model-3','model-4'),
    name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
  )
  p_mimic <- list()
  p_eicu <- list()
  GLM <- list()
  for (i in c(1:4)){
    lx <- c()
    for (j in c(1:i)){
      if (j>=2){
        cr <- c()
        for (z in c(1:length(L[[j]]))){
          f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
          if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
        }
        lx <- c(lx,L[[j]][cr])
      }else{lx <- c(lx,L[[1]])}
    }
    lxy <- c(lx,L[[5]])
    xxx <- x[data$kmeans==lab,lxy]
    r <- Formula2(xxx,y='acute_sepsis ~',n=1)
    x3 <- r[[1]]
    fmla <- r[[2]]
    logfit <- glm(fmla,data=x3,family="binomial")
    GLM[[i]] <- logfit
    # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
    pre <- predict(logfit,type='response')
    roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
    p_mimic[[i]] <- roc1
    roc11 <- cutoff::roc( pre,x3[,L[[5]]])
    result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
    xx_eciu <- x_eicu[data_eicu$kmeans==lab,lx]
    pre_eicu <- predict(logfit,xx_eciu,type='response')
    roc1_eciu <- pROC::roc(x_eicu[data_eicu$kmeans==lab,c(L[[5]])], pre_eicu,ci=TRUE)
    p_eicu[[i]] <- roc1_eciu
    roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[data_eicu$kmeans==lab,c(L[[5]])])
    result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
    if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
    if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
    result[i,'P-value'] <- pp
    result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
  }
  colnames(result) <- c('Acute Death','Features','AUC(MIMIC)','AUC(eICU)','P value')
  R_acu[[lab+2]] <- result
  g1 <- ggroc(p_mimic,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('MIMIC')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('base','tj','com','treat'))+
    theme(legend.position="none")+
    theme(text = element_text(size=6))
  g2 <- ggroc(p_eicu,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('eICU')+
    labs(col = PP[lab+1],size=6)+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
    theme(text = element_text(size=6))
  g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
  g4 <- ggtexttable(R_acu[[lab+2]],rows = NULL,theme = ttheme("light",base_size = 5))
  lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
  lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
  lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
  lay_show(lay123)
  plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
  pdf(file = paste0("./result/sub/LR_acu_",PP[lab+1],".pdf"),
      width = 5,             # 宽
      height = 3            # 高
  )
  f1 <- lay_grid(plots4, lay123)
  dev.off()
}

# 对CCI发生率影响
L[[1]] <- c("Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2")
L[[5]] <- c('cci')
R_cci <- list()
result <- data.frame(
  model <- c('model-1','model-2','model-3','model-4'),
  name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
)
p_mimic <- list()
p_eicu <- list()
GLM <- list()
for (i in c(1:4)){
  lx <- c()
  for (j in c(1:i)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[,lxy]
  r <- Formula2(xxx,y='cci ~',n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  GLM[[i]] <- logfit
  # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  pre <- predict(logfit,type='response')
  roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
  p_mimic[[i]] <- roc1
  roc11 <- cutoff::roc( pre,x3[,L[[5]]])
  result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
  xx_eciu <- x_eicu[,lx]
  pre_eicu <- predict(logfit,xx_eciu,type='response')
  roc1_eciu <- pROC::roc(x_eicu[,c(L[[5]])], pre_eicu,ci=TRUE)
  p_eicu[[i]] <- roc1_eciu
  roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[,c(L[[5]])])
  result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
  if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
  if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
  result[i,'P-value'] <- pp
  result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
}
colnames(result) <- c('CCI','Features','AUC(MIMIC)','AUC(eICU)','P value')
R_cci[[1]] <- result
g1 <- ggroc(p_mimic,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('MIMIC')+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('base','tj','com','treat'))+
  theme(legend.position="none")+
  theme(text = element_text(size=6))
g2 <- ggroc(p_eicu,size = 0.5)+
  labs(col = name[i+1])+
  theme_bw()+
  ggtitle('eICU')+
  labs(col = 'CCI',size=6)+
  theme(plot.title = element_text(family = "serif", #字体
                                  face = "bold",     #字体加粗
                                  color = "black",      #字体颜色
                                  size = 6,          #字体大小
                                  hjust = 0.5,          #字体左右的位置
                                  vjust = 0.5,          #字体上下的高度
                                  angle = 0),             #字体倾斜的角度
        panel.grid.major=element_line(colour=NA),
        # panel.grid = element_blank(),
        # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
        legend.box.background = element_rect(color=NA))+
  scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
  theme(text = element_text(size=6))
g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
g4 <- ggtexttable(R_cci[[1]],rows = NULL,theme = ttheme("light",base_size = 5))
lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
lay_show(lay123)
plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
pdf(file = "./result/sub/LR_cci_all.pdf",
    width = 5,             # 宽
    height = 3            # 高
)
f1 <- lay_grid(plots4, lay123)
dev.off()

for (lab in c(0:3)){
  result <- data.frame(
    model <- c('model-1','model-2','model-3','model-4'),
    name <- c('Clinicians','Clinicians+Demographics','Clinicians+Demographics+Comorbidities','Clinicians+Demographics+Comorbidities+Treatment')
  )
  p_mimic <- list()
  p_eicu <- list()
  GLM <- list()
  for (i in c(1:4)){
    lx <- c()
    for (j in c(1:i)){
      if (j>=2){
        cr <- c()
        for (z in c(1:length(L[[j]]))){
          f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
          if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
        }
        lx <- c(lx,L[[j]][cr])
      }else{lx <- c(lx,L[[1]])}
    }
    lxy <- c(lx,L[[5]])
    xxx <- x[data$kmeans==lab,lxy]
    r <- Formula2(xxx,y='cci ~',n=1)
    x3 <- r[[1]]
    fmla <- r[[2]]
    logfit <- glm(fmla,data=x3,family="binomial")
    GLM[[i]] <- logfit
    # logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
    pre <- predict(logfit,type='response')
    roc1 <- pROC::roc(x3[,L[[5]]], pre,ci=TRUE)
    p_mimic[[i]] <- roc1
    roc11 <- cutoff::roc( pre,x3[,L[[5]]])
    result[i,'Train'] <- paste0(round(roc1$ci[2],2),'(',round(roc1$ci[1],2),',',round(roc1$ci[3],2),')')
    xx_eciu <- x_eicu[data_eicu$kmeans==lab,lx]
    pre_eicu <- predict(logfit,xx_eciu,type='response')
    roc1_eciu <- pROC::roc(x_eicu[data_eicu$kmeans==lab,c(L[[5]])], pre_eicu,ci=TRUE)
    p_eicu[[i]] <- roc1_eciu
    roc11_eciu <- cutoff::roc( pre_eicu,x_eicu[data_eicu$kmeans==lab,c(L[[5]])])
    result[i,'Val'] <- paste0(round(roc1_eciu$ci[2],2),'(',round(roc1_eciu$ci[1],2),',',round(roc1_eciu$ci[3],2),')')
    if(i>=2){if(anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]<0.001){pp <- '<0.001'}else{pp <- anova(GLM[[i-1]],GLM[[i]],test = "Chisq")[2,5]}}else{pp <- 'Ref'}
    if (pp=='<0.001' || pp=='Ref'){pp <- pp}else{pp <- round(pp,3)}
    result[i,'P-value'] <- pp
    result[i,2] <- paste0(result[i,2],'(N=',(length(colnames(x3))-1),')')
  }
  colnames(result) <- c('CCI','Features','AUC(MIMIC)','AUC(eICU)','P value')
  R_cci[[lab+2]] <- result
  g1 <- ggroc(p_mimic,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('MIMIC')+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('base','tj','com','treat'))+
    theme(legend.position="none")+
    theme(text = element_text(size=6))
  g2 <- ggroc(p_eicu,size = 0.5)+
    labs(col = name[i+1])+
    theme_bw()+
    ggtitle('eICU')+
    labs(col = PP[lab+1],size=6)+
    theme(plot.title = element_text(family = "serif", #字体
                                    face = "bold",     #字体加粗
                                    color = "black",      #字体颜色
                                    size = 6,          #字体大小
                                    hjust = 0.5,          #字体左右的位置
                                    vjust = 0.5,          #字体上下的高度
                                    angle = 0),             #字体倾斜的角度
          panel.grid.major=element_line(colour=NA),
          # panel.grid = element_blank(),
          # legend.position = c(0.74,0.2),#更改图例的位置，放至图内部的左上角
          legend.box.background = element_rect(color=NA))+
    scale_colour_hue(labels=c('model-1','model-2','model-3','model-4'))+
    theme(text = element_text(size=6))
  g3 <- ggplot()+theme_bw()+theme(panel.border = element_blank())
  g4 <- ggtexttable(R_cci[[lab+2]],rows = NULL,theme = ttheme("light",base_size = 5))
  lay12 = lay_new(t(matrix(1:2, ncol = 1)),widths = c(6, 8))
  lay4 = lay_new(t(matrix(1:3, ncol = 1)),widths = c(1,9, 1))
  lay123 <- lay_bind_row(lay12, lay4, heights = c(3, 2))
  lay_show(lay123)
  plots4 = lapply(c(1:4), function(x) get(paste0("g", x)))
  pdf(file = paste0("./result/sub/LR_cci_",PP[lab+1],".pdf"),
      width = 5,             # 宽
      height = 3            # 高
  )
  f1 <- lay_grid(plots4, lay123)
  dev.off()
}

# plot forest plots
P_die <- list()
L[[1]] <- c("Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP")
L[[5]] <- c('hospital_expire_flag')
colnames(x) <-gsub('[ ]', '_', colnames(x))
colnames(x_eicu) <-gsub('[ ]', '_', colnames(x_eicu))
for (i in c(1:5)){
  L[[i]] <- gsub('[ ]', '_', L[[i]])
}
lx <- c()
for (j in c(1:4)){
  if (j>=2){
    cr <- c()
    for (z in c(1:length(L[[j]]))){
      f <- summary(glm(as.formula(paste0(L[[5]]," ~ ","`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
      if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
    }
    lx <- c(lx,L[[j]][cr])
  }else{lx <- c(lx,L[[1]])}
}
lxy <- c(lx,L[[5]])
xxx <- x[,lxy]
r <- Formula2(xxx,n=1)
x3 <- r[[1]]
fmla <- r[[2]]
logfit <- glm(fmla,data=x3,family="binomial")
logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
# modelPlot(logfit)
P_die1 <- or_plot2(logreg4,plot_opts=list(xlab("Hospital Mortality, 95% CI"))) #Phenotype "Phenotype D:OR, 95% CI"
Lab <- c('A','B','C','D')
for (lab in c(0:3)){
  lx <- c()
  for (j in c(1:4)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[data$kmeans==lab,lxy]
  r <- Formula2(xxx,n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  P_die[[lab+1]] <- or_plot2(logreg4,plot_opts=list(xlab(paste0("Hospital Mortality of Phenotype ",Lab[lab+1], ":OR,95% CI"))))
}

ggsave(
  filename = paste("./result/sub/forest_die_all.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = P_die1,
  width = 15,             # 宽
  height = 10,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)
f1 <- wrap_plots(P_die, ncol = 2)
ggsave(
  filename = paste("./result/sub/forest_die_ABCD.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 30,             # 宽
  height = 18,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

P_acu <- list()
L[[1]] <- c("Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose")
L[[5]] <- c('acute_sepsis')
for (i in c(1:5)){
  L[[i]] <- gsub('[ ]', '_', L[[i]])
}
lx <- c()
for (j in c(1:3)){
  if (j>=2){
    cr <- c()
    for (z in c(1:length(L[[j]]))){
      f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
      if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
    }
    lx <- c(lx,L[[j]][cr])
  }else{lx <- c(lx,L[[1]])}
}
lxy <- c(lx,L[[5]])
xxx <- x[,lxy]
r <- Formula2(xxx,y='acute_sepsis ~',n=1)
x3 <- r[[1]]
fmla <- r[[2]]
logfit <- glm(fmla,data=x3,family="binomial")
logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
P_acu1 <- or_plot2(logreg4,title_key = 2,plot_opts=list(xlab("Acute Death, 95% CI"))) #Phenotype "Phenotype D:OR, 95% CI"
Lab <- c('A','B','C','D')
for (lab in c(0:3)){
  lx <- c()
  for (j in c(1:3)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[data$kmeans==lab,lxy]
  r <- Formula2(xxx,y='acute_sepsis ~',n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  P_acu[[lab+1]] <- or_plot2(logreg4,title_key = 2,plot_opts=list(xlab(paste0("Acute Death of Phenotype ",Lab[lab+1], ":OR,95% CI"))))
}
ggsave(
  filename = paste("./result/sub/forest_acu_all.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = P_acu1,
  width = 15,             # 宽
  height = 8,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)
f1 <- wrap_plots(P_acu, ncol = 2)
ggsave(
  filename = paste("./result/sub/forest_acu_ABCD.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 30,             # 宽
  height = 16,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)

P_cci <- list()
L[[1]] <- c("Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2") #,"pco2_avg"
L[[5]] <- c('hospital_expire_flag')
for (i in c(1:5)){
  L[[i]] <- gsub('[ ]', '_', L[[i]])
}
lx <- c()
for (j in c(1:4)){
  if (j>=2){
    cr <- c()
    for (z in c(1:length(L[[j]]))){
      f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
      if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
    }
    lx <- c(lx,L[[j]][cr])
  }else{lx <- c(lx,L[[1]])}
}
lxy <- c(lx,L[[5]])
xxx <- x[,lxy]
r <- Formula2(xxx,n=1)
x3 <- r[[1]]
fmla <- r[[2]]
logfit <- glm(fmla,data=x3,family="binomial")
logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
# modelPlot(logreg4)
P_cci1 <- or_plot2(logreg4,title_key = 3,plot_opts=list(xlab("CCI, 95% CI"))) #Phenotype "Phenotype D:OR, 95% CI"
Lab <- c('A','B','C','D')
for (lab in c(0:3)){
  lx <- c()
  for (j in c(1:4)){
    if (j>=2){
      cr <- c()
      for (z in c(1:length(L[[j]]))){
        f <- summary(glm(as.formula(paste0(L[[5]]," ~ ", "`",L[[j]][z],"`")),data=x[,c( L[[j]][z],L[[5]])],family="binomial")) #单因素logstic
        if (f$coefficients[2,'Pr(>|z|)']<=0.05){cr <- c(cr,T)}else{cr <- c(cr,F)}
      }
      lx <- c(lx,L[[j]][cr])
    }else{lx <- c(lx,L[[1]])}
  }
  lxy <- c(lx,L[[5]])
  xxx <- x[data$kmeans==lab,lxy]
  r <- Formula2(xxx,n=1)
  x3 <- r[[1]]
  fmla <- r[[2]]
  logfit <- glm(fmla,data=x3,family="binomial")
  logreg4<-autoReg(logfit,uni=F,threshold=0.05, final=F)#final=T逐步回归
  P_cci[[lab+1]] <- or_plot2(logreg4,title_key = 3,plot_opts=list(xlab(paste0("CCI of Phenotype ",Lab[lab+1], ":OR,95% CI"))))
}
ggsave(
  filename = paste("./result/sub/forest_cci_all.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = P_cci1,
  width = 15,             # 宽
  height = 10,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)
f1 <- wrap_plots(P_cci, ncol = 2)
ggsave(
  filename = paste("./result/sub/forest_cci_ABCD.pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
  plot = f1,
  width = 30,             # 宽
  height = 20,            # 高
  units = "in",          # 单位
  dpi = 300              # 分辨率DPI
)
detach("package:showtext", unload = TRUE)
detach("package:showtextdb", unload = TRUE)