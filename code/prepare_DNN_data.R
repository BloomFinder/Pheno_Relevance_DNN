##Script to prepare Flickr testing dataset for initial classification.

proj_dir <- "~/code/Pheno_Relevance_DNN/"
data_dir <- "/Volumes/HerbariumOffice/Flickr_photos_2009_2016_sample100/"
scratch_dir <- "/Volumes/HerbariumOffice/Flickr_sample100_scratch/"
aws_dir <- "/Users/ian/miniconda2/bin/"

setwd(proj_dir)
library(keras)

metadata <- read.csv("./data/Flickr_sample100_allsites_classed.csv")
metadata$has_flower[metadata$has_flower==""] <- NA
metadata$has_flower[metadata$is_useful_pheno==""] <- NA
metadata <- metadata[complete.cases(cbind(metadata$is_useful_pheno,metadata$has_flower)),]
metadata$filename <- paste("Flickr_",
                           metadata$site,"_",
                           metadata$year,"_",
                           gsub("@","-",metadata$owner),"-",
                           metadata$id,"_z.jpg",sep="")
metadata$filepath <- paste(data_dir,metadata$site,sep="")

##Splits data into training, testing, and validation sets.
set.seed(41)
random <- runif(nrow(metadata),0,1)
train_ind <- random < 0.8
test_ind <- random >= 0.8 & random < 0.9
valid_ind <- random >= 0.9 & random < 1

metadata$split <- as.character(metadata$is_useful_pheno)
metadata$split[metadata$split == "Y"] <- "relevant"
metadata$split[metadata$split == "N"] <- "nonrelevant"

metadata_train <- metadata[train_ind,]
metadata_test <- metadata[test_ind,]
metadata_valid <- metadata[valid_ind,]

##Copies files to different folders depending on their tag.
dir.create(paste(scratch_dir,"training",sep=""))
dir.create(paste(scratch_dir,"testing",sep=""))
dir.create(paste(scratch_dir,"validation",sep=""))

split_photos <- function(metadata=metadata_train,
                         split_field="split",
                         filename_field="filename",
                         filepath_field="filepath",
                         output_dir=paste(scratch_dir,"training/",sep=""),
                         overwrite=TRUE,
                         missing_fail=FALSE){
    
    split_options <- unique(metadata[, colnames(metadata) == split_field])
    for(i in 1:length(split_options)){
      metadata_split <- metadata[metadata[,colnames(metadata) == split_field] == split_options[i],]
      out <- paste(output_dir,split_options[i],sep="")
      if(!(dir.exists(out)) | overwrite == TRUE){
        dir.create(path=out)
      }else{
        print(paste("Directory",outpath,"exists, skipping..."))
        next
      }
    }
    frompaths <- paste(metadata[,colnames(metadata) == filepath_field],"/",
                       metadata[,colnames(metadata) == filename_field],sep="")
    
    topaths <- paste(output_dir,
                     metadata[,colnames(metadata) == split_field],"/",
                     metadata[,colnames(metadata) == filename_field],
                     sep="")

    if(all(file.exists(frompaths))){
      print(paste("Copying", length(metadata[,colnames(metadata) == filepath_field]),"files"))
      file.copy(from=frompaths,to=topaths,overwrite=overwrite)
    }else if(missing_fail==FALSE){
      print(paste("Copying, but some input files do not exist!"))
      print(paste("Missing files: \n",
                  frompaths[!file.exists(frompaths)]))
      file.copy(from=frompaths,to=topaths,overwrite=overwrite)
    }else{
      print(paste("Some input files do not exist!"))
      print(paste("Missing files: \n",
                  frompaths[!file.exists(frompaths)]))
    }
    return(metadata[file.exists(frompaths),])
}

train_meta <- split_photos(metadata=metadata_train,
                   split_field="split",
                   filename_field="filename",
                   filepath_field="filepath",
                   output_dir=paste(scratch_dir,"training/",sep=""),
                   overwrite=TRUE,
                   missing_fail=FALSE)

test_meta <- split_photos(metadata=metadata_test,
                 split_field="split",
                 filename_field="filename",
                 filepath_field="filepath",
                 output_dir=paste(scratch_dir,"testing/",sep=""),
                 overwrite=TRUE,
                 missing_fail=FALSE)

valid_meta <-split_photos(metadata=metadata_valid,
                 split_field="split",
                 filename_field="filename",
                 filepath_field="filepath",
                 output_dir=paste(scratch_dir,"validation/",sep=""),
                 overwrite=TRUE,
                 missing_fail=FALSE)
labels_relev <- list(training=train_meta,testing=test_meta,validation=valid_meta)
saveRDS(labels_relev, paste(scratch_dir,"labels_relev.Rdata", sep=""))


##Does the same thing for flower / nonflower.
metadata$split2 <- as.character(metadata$has_flower)
metadata$split2[metadata$split2 == "Y"] <- "flower"
metadata$split2[metadata$split2 == "N"] <- "nonflower"

metadata_train2 <- metadata[train_ind,]
metadata_test2 <- metadata[test_ind,]
metadata_valid2 <- metadata[valid_ind,]


train_meta2 <- split_photos(metadata=metadata_train2,
                           split_field="split2",
                           filename_field="filename",
                           filepath_field="filepath",
                           output_dir=paste(scratch_dir,"training/",sep=""),
                           overwrite=TRUE,
                           missing_fail=FALSE)

test_meta2 <- split_photos(metadata=metadata_test2,
                          split_field="split2",
                          filename_field="filename",
                          filepath_field="filepath",
                          output_dir=paste(scratch_dir,"testing/",sep=""),
                          overwrite=TRUE,
                          missing_fail=FALSE)

valid_meta2 <-split_photos(metadata=metadata_valid2,
                          split_field="split2",
                          filename_field="filename",
                          filepath_field="filepath",
                          output_dir=paste(scratch_dir,"validation/",sep=""),
                          overwrite=TRUE,
                          missing_fail=FALSE)
labels_flwr <- list(training=train_meta2,testing=test_meta2,validation=valid_meta2)
saveRDS(labels_flwr, paste(scratch_dir,"labels_flower.Rdata", sep=""))

##Zips up data and uploads to S3.
tar_data_cmd <- paste("cd ",data_dir, " && tar -zcvf ../Flickr100.tar.gz ", scratch_dir,
                      sep="")
system(tar_data_cmd,wait=TRUE)

aws_upl_cmd <- paste("cd ", data_dir, " && ",aws_dir,"aws s3 cp ../Flickr100.tar.gz ",
                     "s3://sdmdata/sample_photos",sep="")
system(aws_upl_cmd,wait=TRUE)