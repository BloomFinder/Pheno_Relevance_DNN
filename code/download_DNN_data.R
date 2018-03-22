##Script to download selected photos from Flickr for traial.
##Author:Ian Breckheimer
##27 October 2017

##Sets up workspace
library(readr)
library(dplyr)
library(foreach)
library(doParallel)
setwd("~/code/Pheno_Relevance_DNN/")

##Reads in data.
metadat <- read_csv("./data/Flickr_sample100_allsites_classed.csv")
metadat$URL <- paste("http://flickr.com/photos/",metadat$owner,"/",metadat$id,sep="")
metadat$owner_id <- paste(metadat$owner,metadat$id)

photo_loc <- "/Volumes/HerbariumOffice/Flickr_photos_2009_2016_sample100"

##Gets a list of site names
sites <- unique(metadat$site)

##Creates folders for sites if they don't already exist.
sites_col <- paste(sites,collapse=" ")
mkdir_cmd <- paste("cd",photo_loc,"&& mkdir ",sites_col,sep=" ")
system(mkdir_cmd)

df