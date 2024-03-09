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

###-----Modeling vasoactive drugs and hospital mortality with RCS and PSM-------###
# Note: It is recommended to re-open Rstudio when running the following code to avoid bugs.
# functions part
datatime <- function (data=data, format="%Y-%m-%d %H:%M:%S"){
  a11 <- strsplit(data, " ")[[1]]
  a12 <- strsplit(a11[1], "/")[[1]]
  a13 <- paste0(a12[3],'-',a12[2],'-',a12[1],' ',a11[2])
  time <- strptime(a13, format)
  return(time)
}
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
or_plot3 <- function(OR){
  ORR1 <- c()
  for (i in c(1:(length(OR[,1])-1))){
    if(OR[i,2]<1 && OR[i+1,2]>=1){
      ORR1 <- c(ORR1,c(OR[i+1,1]))
    }
    if(OR[i,2]>=1 && OR[i+1,2]<1){
      ORR1 <- c(ORR1,c(OR[i+1,1]))
    }
  }
  return(ORR1)
}
rcs_logistic.lshap1 <- function (data, knot, y, x, covs, prob, ...) 
{
  pacman::p_load(rms, ggplot2, survminer, survival, dplyr, 
    patchwork, Cairo)
  call <- match.call()
  data <- as.data.frame(data)
  y <- y
  x <- x
  if (missing(covs)) {
    covs = NULL
    indf <- dplyr::select(data, y, x)
  }
  else {
    assign("covs", covs)
    indf <- dplyr::select(data, y, x, covs)
  }
  indf[, "y"] <- indf[, y]
  indf[, "x"] <- indf[, x]
  sum(!complete.cases(indf[, c(y, x)]))
  indf <- indf[complete.cases(indf[, c(y, x)]), ]
  dd <- NULL
  dd <<- rms::datadist(indf)
  old <- options()
  on.exit(options(old))
  options(datadist = "dd")
  aics <- NULL
  if (is.null(covs)) {
    formula <- paste0("y~ rcs(x, ", knot, ")", paste0(covs, 
      collapse = " + "))
  }
  else {
    formula <- paste0("y~ rcs(x, ", knot, ")", " + ", paste0(covs, 
      collapse = " + "))
  }
  model <- rms::lrm(as.formula(formula), data = indf, x = TRUE, 
    se.fit = TRUE, tol = 1e-25)
  model.logistic <- model
  anova(model)
  pvalue_all <- anova(model)[1, 3]
  pvalue_nonlin <- round(anova(model)[2, 3], 3)
  pre.model <- rms::Predict(model.logistic, x, fun = exp, 
    type = "predictions", ref.zero = T, conf.int = 0.95, 
    digits = 2)
  Q20 <- quantile(indf$x, probs = seq(0, 1, 0.05))
  pre.model <- rms::Predict(model.logistic, x, fun = exp, 
    type = "predictions", ref.zero = F, conf.int = 0.95, 
    digits = 2)
  ORR1 <- or_plot3(pre.model)
  newdf1 <- as.data.frame(dplyr::select(pre.model, x, yhat, 
    lower, upper))
  colnames(newdf1) <- c("x", "y", "lower", "upper")
  min(newdf1[, "x"])
  max(newdf1[, "x"])
  xmin <- min(newdf1[, "x"])
  xmax <- max(newdf1[, "x"])
  min(newdf1[, "lower"])
  max(newdf1[, "upper"])
  ymax1 <- ceiling(max(newdf1[, "upper"]))
  newdf2 <- indf[indf[, "x"] >= xmin & indf[, "x"] <= xmax, 
    ]
  breaks <- seq(xmin, xmax, length = 20)
  h <- hist(newdf2[, "x"], breaks = breaks, right = TRUE)
  max(h[["counts"]]/sum(h[["counts"]]))
  
  newdf3 <- data.frame(x = h[["mids"]], freq = h[["counts"]], 
    pct = h[["counts"]]/sum(h[["counts"]]))
  ymax2 <- ceiling(max(newdf3$pct*100))
  
  freq <- cut(newdf2[, "x"], breaks = breaks, dig.lab = 6, 
    right = TRUE)
  as.data.frame(table(freq))
  scale_factor <- ymax2/ymax1*1.1
  xtitle <- x
  OOOR <- "None"
  if (!is.null(ORR1)){
    OOOR <- paste0(round(ORR1,4), collapse= ", ")
  }
  ytitle1 <- paste0("OR where the cutoff for ", x, " is ",OOOR)
  ytitle2 <- "Percentage of Population (%)"
  offsetx1 <- (xmax - xmin) * 0.02
  offsety1 <- ymax1 * 0.02
  labelx1 <- xmin + (xmax - xmin) * 0.15
  labely1 <- ymax1 * 0.8
  label1 <- paste0("Estimation", "\n", "95% CI")
  labelx2 <- xmin + (xmax - xmin) * 0.7
  labely2 <- ymax1 * 0.8
  label2 <- paste0("P-overall = ", ifelse(pvalue_all < 0.001, 
    "< 0.001", sprintf("%.3f", pvalue_all)), "\nP-non-linear = ", 
    ifelse(pvalue_nonlin < 0.001, "< 0.001", sprintf("%.3f", 
      pvalue_nonlin)))
  plot.lshap.type2 <- ggplot2::ggplot() + geom_bar(data = newdf3,
    aes(x = x, y = pct * 100/scale_factor), stat = "identity",
    width = (xmax - xmin)/(length(breaks) - 1), fill = "#f9f7f7",
    color = "grey") + geom_hline(yintercept = 1, size = 1,
    linetype = 2, color = "grey") + geom_ribbon(data = newdf1,
    aes(x = x, ymin = lower, ymax = upper), fill = "#e23e57",
    alpha = 0.1) + geom_line(data = newdf1, aes(x = x, y = lower),
    linetype = 0, color = "#ff9999", size = 0.8) + geom_line(data = newdf1,
    aes(x = x, y = upper), linetype = 0, color = "#ff9999",
    size = 0.8) + geom_line(data = newdf1, aes(x = x, y = y),
    color = "#e23e57", size = 1) + geom_segment(aes(x = c(labelx1 -
    offsetx1 * 5, labelx1 - offsetx1 * 5), xend = c(labelx1 -
    offsetx1, labelx1 - offsetx1), y = c(labely1 + offsety1,
    labely1 - offsety1), yend = c(labely1 + offsety1, labely1 -
    offsety1)), linetype = c(1, 2), color = c("#e23e57",
    "black")) + geom_text(aes(x = labelx1, y = labely1,
    label = label1), hjust = 0) + geom_text(aes(x = labelx2,
    y = labely2, label = label2), hjust = 0) + scale_x_continuous(xtitle,
    expand = c(0, 0.00)) + scale_y_continuous(ytitle1,expand = c(0, 0.00), 
     sec.axis = sec_axis(ytitle2,
      trans = ~. * scale_factor, )) + theme_bw() + theme(axis.line = element_line(),
    panel.grid = element_blank(), panel.border = element_blank())
  if (!is.null(ORR1)){
    plot.lshap.type2 <- plot.lshap.type2 + geom_point(aes(x = ORR1, y = 1), color = "#e23e57", size = 2)
  }
  fig.lshapall <- plot.lshap.type2
  return(fig.lshapall)
}
# MIMIC
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
x[,62] <- data$gender
x[x[,62]=="M",62] <- 1
x[x[,62]=="F",62] <- 0
x[,62] <- factor(x[,62], levels=c(0, 1))
summary(x)
for (i in c(89:96,78,79)){
  x[,i] <- factor(x[,i], levels=c(0, 1))
}
data_g <- data.frame()
for (i in c(1:33177)) {
  if (is.na(data[i,'dopamine']) || data[i,'dopamine']>quantile(data$dopamine, 0.99, na.rm=TRUE)){
    data_g[i,'dopamine'] <- 0
  }
  else {
    data_g[i,'dopamine'] <- 1
  }
  if (is.na(data[i,'epinephrine']) || data[i,'epinephrine']>quantile(data$epinephrine, 0.99, na.rm=TRUE)){
    data_g[i,'epinephrine'] <- 0
  }
  else {
    data_g[i,'epinephrine'] <- 1
  }
  if (is.na(data[i,'norepinephrine']) || data[i,'norepinephrine']>quantile(data$norepinephrine, 0.99, na.rm=TRUE)){
    data_g[i,'norepinephrine'] <- 0
  }
  else {
    data_g[i,'norepinephrine'] <- 1
  }
  if (is.na(data[i,'phenylephrine']) || data[i,'phenylephrine']>quantile(data$phenylephrine, 0.99, na.rm=TRUE)){
    data_g[i,'phenylephrine'] <- 0
  }
  else {
    data_g[i,'phenylephrine'] <- 1
  }
  if (is.na(data[i,'vasopressin']) || data[i,'vasopressin']>quantile(data$vasopressin, 0.99, na.rm=TRUE)){
    data_g[i,'vasopressin'] <- 0
  }
  else {
    data_g[i,'vasopressin'] <- 1
  }
  if (is.na(data[i,'dobutamine']) || data[i,'dobutamine']>quantile(data$dobutamine, 0.99, na.rm=TRUE)){
    data_g[i,'dobutamine'] <- 0
  }
  else {
    data_g[i,'dobutamine'] <- 1
  }
  if (is.na(data[i,'milrinone']) || data[i,'milrinone']>quantile(data$milrinone, 0.99, na.rm=TRUE)){
    data_g[i,'milrinone'] <- 0
  }
  else {
    data_g[i,'milrinone'] <- 1
  }
  if (data[i,'starttime'] != ''){
    x[i,'time'] <- as.integer(difftime(datatime(data[i,'starttime']),datatime(data[i,'admittime']),units = 'mins'))/1440
  }
}

x$dopamine <- data$dopamine
x$epinephrine <- data$epinephrine
x$norepinephrine <- data$norepinephrine
x$phenylephrine <- data$phenylephrine
x$vasopressin <- data$vasopressin
x$dobutamine <- data$dobutamine
x$milrinone <- data$milrinone
xx <- x[,c(1:63,78,79,89:96,87)]
title1 <- c('All Vasoactive Drugs','Dopamine','Epinephrine','Norepinephrine','Phenylephrine','Vasopressin','Dobutamine','Milrinone')
fig <- list()
fig1 <- list()
fig_smd <- list()
E <- c('A','B','C','D','ALL')
for (j in c(0:4)){
  for (i in c(1:8)){
    xxx <- xx
    strata <- colnames(xxx)[74]
    xxx$timing <- x$time
    if (j==4){if (i == 1){xxx <- xxx[data$starttime != '',]}else{xxx <- xxx[data_g[,i-1]==1,]}}else{
      if (i == 1){xxx <- xxx[data$starttime != '' & data$kmeans==j,]}else{xxx <- xxx[data_g[,i-1]==1 & data$kmeans==j,]}}
    r <- Formula2(xxx,n=2)
    x3 <- r[[1]]
    fmla <- r[[2]]
    #判断y是否全为0或1
    if(all(x3[,'hospital_expire_flag'] == x3[1,'hospital_expire_flag'])){a1 <- T}else{a1 <- F}
    if(a1){fig[[i]] <- ggplot()+theme_bw()}else
    {
      matching = matchit(fmla,data=x3, method="nearest", ratio=1)
      Matched_data = match.data(matching)
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=2)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=3)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){fig[[i]] <- ggplot()+theme_bw()}else #判断数量是否够
      {
        DD <- Matched_data[,c('hospital_expire_flag','timing')]
        dd <- datadist(DD)
        options(datadist="dd")
        fig[[i]] <- rcs_logistic.lshap1(data=DD, knot=3,y = "hospital_expire_flag",x = "timing", prob=0.1)+
          labs(title = title1[i])
      }
    }
    f1 <- wrap_plots(fig, ncol = 4)
    ggsave(
      filename = paste("./result/sub/rcs_time_mimic",E[j+1],".pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
      plot = f1,
      width = 28,             # 宽
      height = 10,            # 高
      units = "in",          # 单位
      dpi = 800              # 分辨率DPI
    )
  }
}

title1 <- c('dopamine','epinephrine','norepinephrine','phenylephrine','vasopressin','dobutamine','milrinone')
title <- c('Dopamine','Epinephrine','Norepinephrine','Phenylephrine','Vasopressin','Dobutamine','Milrinone')
fig <- list()
fig1 <- list()
for (j in c(0:4)){
  for (i in c(1:7)){
    xxx <- xx
    xxx$xx <- x[,title1[i]]
    if (j==4){xxx <- xxx[data_g[,i]==1,]}else{xxx <- xxx[data_g[,i]==1 & data$kmeans==j,]}
    r <- Formula2(xxx,n=2)
    x3 <- r[[1]]
    fmla <- r[[2]]
    #判断y是否全为0或1
    if(all(x3[,'hospital_expire_flag'] == x3[1,'hospital_expire_flag'])){a1 <- T}else{a1 <- F}
    if(a1){fig[[i]] <- ggplot()+theme_bw()}else
    {
      matching = matchit(fmla,data=x3, method="nearest", ratio=1)
      Matched_data = match.data(matching)
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=2)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=3)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){fig[[i]] <- ggplot()+theme_bw()}else #判断数量是否够
      {
        DD <- Matched_data[,c('hospital_expire_flag','xx')]
        colnames(DD) <- c('hospital_expire_flag','dosage')
        fig[[i]] <- rcs_logistic.lshap1(data=DD, knot=3,y = "hospital_expire_flag",x = "dosage", prob=0.1)+
          labs(title = title[i])
      }
    }
    f1 <- wrap_plots(fig, ncol = 4)
    ggsave(
      filename = paste("./result/sub/rcs_drug_mimic",E[j+1],".pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
      plot = f1,
      width = 28,             # 宽
      height = 10,            # 高
      units = "in",          # 单位
      dpi = 800              # 分辨率DPI
    )
  }
}

##EICU
data <- read.csv("./data/sepsis3_eicu_kmeans_drop_p2.csv")
x = data[,c(22:82,16,14,83:90,105:118,119:120,101:104,121,122,15,123)]
for (i in c(62,64:71,86,87)){
  x[is.na(x[,i]),i] <- 0
}
for (i in c(62,64:71,86,87)){
  x[,i] <- factor(x[,i], levels=c(0,1))
}
colnames(x)[94] <- 'hospital_expire_flag'
data_g <- data.frame()
for (i in c(1:21496)) {
  if (is.na(data[i,'dopamine']) || data[i,'dopamine']>quantile(data$dopamine, 0.99, na.rm=TRUE)){
    data_g[i,'dopamine'] <- 0
  }
  else {
    data_g[i,'dopamine'] <- 1
  }
  if (is.na(data[i,'epinephrine']) || data[i,'epinephrine']>quantile(data$epinephrine, 0.99, na.rm=TRUE)){
    data_g[i,'epinephrine'] <- 0
  }
  else {
    data_g[i,'epinephrine'] <- 1
  }
  if (is.na(data[i,'norepinephrine']) || data[i,'norepinephrine']>quantile(data$norepinephrine, 0.99, na.rm=TRUE)){
    data_g[i,'norepinephrine'] <- 0
  }
  else {
    data_g[i,'norepinephrine'] <- 1
  }
  if (is.na(data[i,'phenylephrine']) || data[i,'phenylephrine']>quantile(data$phenylephrine, 0.99, na.rm=TRUE)){
    data_g[i,'phenylephrine'] <- 0
  }
  else {
    data_g[i,'phenylephrine'] <- 1
  }
  if (is.na(data[i,'vasopressin']) || data[i,'vasopressin']>quantile(data$vasopressin, 0.99, na.rm=TRUE)){
    data_g[i,'vasopressin'] <- 0
  }
  else {
    data_g[i,'vasopressin'] <- 1
  }
  if (is.na(data[i,'dobutamine']) || data[i,'dobutamine']>quantile(data$dobutamine, 0.99, na.rm=TRUE)){
    data_g[i,'dobutamine'] <- 0
  }
  else {
    data_g[i,'dobutamine'] <- 1
  }
  if (is.na(data[i,'milrinone']) || data[i,'milrinone']>quantile(data$milrinone, 0.99, na.rm=TRUE)){
    data_g[i,'milrinone'] <- 0
  }
  else {
    data_g[i,'milrinone'] <- 1
  }
  if (is.na(data[i,'starttime'])){
    x[i,'time'] <- data[i,'starttime']
  }
  else {
    x[i,'time'] <- (data[i,'starttime']-data[i,'hospitaladmitoffset'])/1440
  }
}

x$dopamine <- data$dopamine
x$epinephrine <- data$epinephrine
x$norepinephrine <- data$norepinephrine
x$phenylephrine <- data$phenylephrine
x$vasopressin <- data$vasopressin
x$dobutamine <- data$dobutamine
x$milrinone <- data$milrinone
xx <- x[,c(1:71,86,87,94)]
summary(xx)
title1 <- c('All Vasoactive Drugs','Dopamine','Epinephrine','Norepinephrine','Phenylephrine','Vasopressin','Dobutamine','Milrinone')
fig <- list()
fig1 <- list()
E <- c('A','B','C','D','ALL')
for (j in c(0:4)){
  for (i in c(1:8)){
    xxx <- xx
    xxx$timing <- x$time
    if (j==4){if (i == 1){xxx <- xxx[!is.na(data$starttime),]}else{xxx <- xxx[data_g[,i-1]==1,]}}else{
      if (i == 1){xxx <- xxx[!is.na(data$starttime) & data$kmeans==j,]}else{xxx <- xxx[data_g[,i-1]==1 & data$kmeans==j,]}}
    r <- Formula2(xxx,n=2)
    x3 <- r[[1]]
    fmla <- r[[2]]
    #判断y是否全为0或1
    if(all(x3[,'hospital_expire_flag'] == x3[1,'hospital_expire_flag'])){a1 <- T}else{a1 <- F}
    if(a1){fig[[i]] <- ggplot()+theme_bw()
    fig1[[i]] <- ggplot()+theme_bw()}else
    {
      matching = matchit(fmla,data=x3, method="nearest", ratio=1)
      Matched_data = match.data(matching)
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=2)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=3)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){fig[[i]] <- ggplot()+theme_bw()
      fig1[[i]] <- ggplot()+theme_bw()}else #判断数量是否够
      {
        DD <- Matched_data[,c('hospital_expire_flag','timing')]
        dd <- datadist(DD)
        fig[[i]] <- rcs_logistic.lshap1(data=DD, knot=3,y = "hospital_expire_flag",x = "timing", prob=0.1)+
          labs(title = title1[i])
      }
    }
    f1 <- wrap_plots(fig, ncol = 4)
    ggsave(
      filename = paste("./result/sub/rcs_time_eicu",E[j+1],".pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
      plot = f1,
      width = 28,             # 宽
      height = 10,            # 高
      units = "in",          # 单位
      dpi = 800              # 分辨率DPI
    )
  }
}

title1 <- c('dopamine','epinephrine','norepinephrine','phenylephrine','vasopressin','dobutamine','milrinone')
title <- c('Dopamine','Epinephrine','Norepinephrine','Phenylephrine','Vasopressin','Dobutamine','Milrinone')
fig <- list()
fig1 <- list()
for (j in c(0:4)){
  for (i in c(1:7)){
    xxx <- xx
    xxx$xx <- x[,title1[i]]
    if (j==4){xxx <- xxx[data_g[,i]==1,]}else{xxx <- xxx[data_g[,i]==1 & data$kmeans==j,]}
    r <- Formula2(xxx,n=2)
    x3 <- r[[1]]
    fmla <- r[[2]]
    #判断y是否全为0或1
    if(all(x3[,'hospital_expire_flag'] == x3[1,'hospital_expire_flag'])){a1 <- T}else{a1 <- F}
    if(a1){fig[[i]] <- ggplot()+theme_bw()
    fig1[[i]] <- ggplot()+theme_bw()}else
    {
      matching = matchit(fmla,data=x3, method="nearest", ratio=1)
      Matched_data = match.data(matching)
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=2)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){
        matching = matchit(fmla,data=x3, method="nearest", ratio=3)
        Matched_data = match.data(matching)
      }
      if (length(Matched_data[,1])<=15){fig[[i]] <- ggplot()+theme_bw()
      fig1[[i]] <- ggplot()+theme_bw()}else #判断数量是否够
      {
        DD <- Matched_data[,c('hospital_expire_flag','xx')]
        colnames(DD) <- c('hospital_expire_flag','dosage')
        fig[[i]] <- rcs_logistic.lshap1(data=DD, knot=3,y = "hospital_expire_flag",x = "dosage", prob=0.1)+
          labs(title = title[i])
      }
    }
    f1 <- wrap_plots(fig, ncol = 4)
    ggsave(
      filename = paste("./result/sub/rcs_drug_eicu",E[j+1],".pdf"), # 保存的文件名称。通过后缀来决定生成什么格式的图片
      plot = f1,
      width = 28,             # 宽
      height = 10,            # 高
      units = "in",          # 单位
      dpi = 800              # 分辨率DPI
    )
  }
}